# seekingalpha_twitter_stock_analyzer


This tool performs sentiment analysis and extracts insights from financial articles and tweets for a given stock ticker. It utilizes various APIs and machine learning models to provide a comprehensive analysis.

## Features

- Twitter sentiment analysis for a given stock ticker
- Seeking Alpha article retrieval and analysis
- Price target extraction from financial articles
- RAG (Retrieval-Augmented Generation) pipeline for in-depth article analysis
- Multi-threaded processing for improved performance

## Dependencies

To run this project, you'll need the following Python libraries:
requests
llama_index
dspy
beautifulsoup4
joblib
langchain_google_genai
cleanlab_studio
pydantic
transformers
torch

You'll also need API keys for the following services:
- Google AI (Gemini)
- RapidAPI (for Seeking Alpha and Twitter)
- Cleanlab Studio

## Setup

1. Clone this repository
2. Install the required dependencies:
pip install -r requirements.txt

3. Set up your API keys as environment variables or update them in the script:
- `gemini_api_key`
- RapidAPI key
- Cleanlab Studio API key

## Usage

Run the main script:
python stock_analysis_tool.py
By default, the script analyzes AMD stock. You can modify the `ticker` variable in the main function to analyze a different stock.

## Main Components

1. **Twitter Sentiment Analysis**: Extracts recent tweets about the stock and performs sentiment analysis.

2. **Seeking Alpha Article Analysis**: 
   - Retrieves recent articles from Seeking Alpha
   - Cleans and processes the article text
   - Extracts price targets
   - Performs sentiment analysis on each article

3. **RAG Pipeline**: Uses a combination of embedding models and language models to provide in-depth analysis of the articles, including bullish and bearish arguments, and overall sentiment.

## Output

The script provides:
- Twitter sentiment breakdown
- Mean price target from Seeking Alpha articles
- Sentiment analysis for each Seeking Alpha article
- In-depth analysis of bullish and bearish arguments from the RAG pipeline

## Notes

- The script uses multithreading to improve performance, especially for API calls and model inferences.
- You can toggle between using saved article data or fetching new data by modifying the `use_saved_file` variable.
- This tool utilizes the free tier of the Gemini API. As a result, you may encounter rate limit errors if you make too many requests in a short period. If this happens, you may need to wait before running the script again or consider upgrading to a paid tier for higher usage limits.

## Disclaimer

This tool is for educational and research purposes only. Always do your own due diligence before making investment decisions.

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/yourusername/stock-analysis-tool/issues) if you want to contribute.

## License

[MIT](https://choosealicense.com/licenses/mit/)



