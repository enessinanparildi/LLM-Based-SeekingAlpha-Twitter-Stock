from flask import Flask, render_template, request, jsonify, Response
from datetime import datetime, timedelta
from main import load_article_data, twitter_data_extractor, twitter_sentiment_breakdown, save_new_article_data, seeking_alpha_article_breakdown, extract_price_target, setup_RAG_pipeline, stock_overview
from joblib import Parallel, delayed
import dspy
import traceback  # Add this import for detailed error tracking
import time
import json

app = Flask(__name__)

gemini_api_key = 'AIzaSyCE9RhflGUGJpNg6DsLo2obVCGDP_HHVAo'
gemini = dspy.LM(model="gemini/gemini-1.5-pro", api_key=gemini_api_key)
dspy.configure(lm=gemini)


def send_event(event_name, data):
    """Helper function to format SSE events"""
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"

@app.route('/', methods=['GET'])
def home():
    """Render the home page with the analysis form"""
    return render_template('index.html')

@app.route('/analyze_stream', methods=['GET'])
def analyze_stream():
    """Stream analysis results"""
    ticker = request.args.get('ticker', '').strip().upper()
    time_window = int(request.args.get('time_window', 30))

    def generate():
        try:
            # Stock Overview
            yield send_event('status', {'message': 'Getting stock overview...'})
            try:
                overview = stock_overview(ticker)
                yield send_event('stock_overview', {'result': overview})
                time.sleep(5)  # Sleep after LLM call
            except Exception as e:
                print(f"Stock overview error: {str(e)}")
                yield send_event('status', {'message': f'Stock overview failed: {str(e)}'})

            # Twitter Analysis
            yield send_event('status', {'message': 'Analyzing Twitter sentiment...'})
            try:
                tweets = twitter_data_extractor(stock_ticker=ticker)
                time.sleep(5)  # Sleep after API call
                
                sentiment = twitter_sentiment_breakdown(tweets, ticker=ticker)
                yield send_event('twitter_sentiment', {'result': sentiment})
                time.sleep(5)  # Sleep after LLM call
            except Exception as e:
                print(f"Twitter analysis error: {str(e)}")
                yield send_event('status', {'message': f'Twitter analysis failed: {str(e)}'})
            
            # Article Analysis
            yield send_event('status', {'message': 'Loading article data...'})
            try:
                # First try to load existing data
                try:
                    article_data = load_article_data(ticker)
                except FileNotFoundError:
                    yield send_event('status', {'message': 'No saved data found, fetching new articles...'})
                    save_new_article_data(ticker)
                    article_data = load_article_data(ticker)
                time.sleep(3)
            except Exception as e:
                print(f"Article loading error: {str(e)}")
                yield send_event('error', {'message': f'Failed to load articles: {str(e)}'})
                return
            
            # Analyze articles one by one
            yield send_event('status', {'message': 'Analyzing articles...'})
            for idx, article in enumerate(article_data):
                try:
                    analysis = seeking_alpha_article_breakdown(article, ticker)
                    yield send_event('article_analysis', {
                        'index': idx,
                        'result': analysis
                    })
                    time.sleep(40)  # Increased sleep between article analyses
                except Exception as e:
                    print(f"Article analysis error for article {idx}: {str(e)}")
                    continue
            
            # Price Targets
            yield send_event('status', {'message': 'Extracting price targets...'})
            price_targets = []
            for article in article_data:
                try:
                    pt = extract_price_target(article)
                    if pt != '-1':
                        price_targets.append(float(pt))
                        yield send_event('price_target', {
                            'targets': price_targets,
                            'average': sum(price_targets) / len(price_targets) if price_targets else 0
                        })
                    time.sleep(40)  # Increased sleep between price target extractions
                except Exception as e:
                    print(f"Price target extraction error: {str(e)}")
                    continue
            
            # RAG Analysis
            try:
                yield send_event('status', {'message': 'Setting up RAG pipeline...'})
                query_engine = setup_RAG_pipeline(article_data)
                time.sleep(40)
                
                yield send_event('status', {'message': 'Analyzing bullish arguments...'})
                bullish = query_engine.query("What are the main bullish arguments mentioned in the articles?").response
                yield send_event('rag_bullish', {'result': bullish})
                time.sleep(40)  # Sleep after RAG query
                
                yield send_event('status', {'message': 'Analyzing bearish arguments...'})
                bearish = query_engine.query("What are the main bearish arguments mentioned in the articles?").response
                yield send_event('rag_bearish', {'result': bearish})
                time.sleep(40)  # Sleep after RAG query
                
                yield send_event('status', {'message': 'Analyzing catalysts...'})
                catalysts = query_engine.query("What are the key upcoming catalysts mentioned in the articles?").response
                yield send_event('rag_catalysts', {'result': catalysts})
                time.sleep(40)  # Sleep after RAG query
            except Exception as e:
                print(f"RAG analysis error: {str(e)}")
                yield send_event('status', {'message': f'RAG analysis failed: {str(e)}'})
            
            yield send_event('complete', {'message': 'Analysis complete!'})
            
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Fatal error: {str(e)}\n{error_details}")
            yield send_event('error', {'message': f'An error occurred: {str(e)}'})

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)