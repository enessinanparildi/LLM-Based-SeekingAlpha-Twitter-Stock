 Import necessary libraries and modules

"""
This module provides functions for analyzing stock-related data from various sources
including Seeking Alpha and Twitter. It includes tools for fetching articles,
extracting text, performing sentiment analysis, and setting up a RAG (Retrieval-Augmented Generation) pipeline.

Requirements:
- Python libraries: requests, BeautifulSoup, joblib, llama_index, dspy
- API keys for Seeking Alpha, Twitter, and Gemini
"""


# Import necessary libraries and modules
import requests
from llama_index.llms.gemini import Gemini
from llama_index.core import PromptTemplate
import dspy
from bs4 import BeautifulSoup
import joblib
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from cleanlab_studio import Studio
from joblib import Parallel, delayed
import time

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
from llama_index.agent.openai import OpenAIAgent
from dsp.modules import GoogleVertexAI
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent

from dspy import InputField, OutputField, Signature
from dspy.functional import TypedChainOfThought
from pydantic import BaseModel

gemini_api_key = 'gemini_api_key_here'
tlm_api = 'tlm_api_key_here'
rapidapi_key = 'rapid_api_key'


# Set environment variables for API keys and configurations
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Define safety settings for the Gemini model
SAFE = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]



# Function to get Seeking Alpha article IDs for a given stock ticker
def get_seeking_alpha_article_ids(stock_ticker="sofi", number_of_items=str(10)):
    """
    Fetches article IDs from Seeking Alpha for a given stock ticker.

    :param stock_ticker: Stock symbol to search for (default: "sofi")
    :param number_of_items: Number of articles to retrieve (default: "10")
    :return: List of article IDs
    """
    url = "https://seeking-alpha.p.rapidapi.com/analysis/v2/list"

    querystring = {"id": stock_ticker, "size": number_of_items, "number": "1"}

    headers = {
        "x-rapidapi-key": "799f09a0f2msh88cbc5a33cc1067p1d87e3jsne229a1e7c05c",
        "x-rapidapi-host": "seeking-alpha.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()['data']
    id_list = []
    for elem in data:
        id_list.append(elem['id'])

    return id_list


# Function to extract Twitter data for a given stock ticker
def twitter_data_extractor(stock_ticker="sofi"):
    """
    Extracts recent tweets related to a given stock ticker.

    :param stock_ticker: Stock symbol to search for (default: "sofi")
    :return: String containing formatted tweets
    """

    url = "https://twitter154.p.rapidapi.com/search/search"

    querystring = {"query": "$" + stock_ticker, "section": "top", "min_likes": "5", "limit": "20",
                   "start_date": "2023-06-01", "language": "en"}
    headers = {
        "x-rapidapi-key": "799f09a0f2msh88cbc5a33cc1067p1d87e3jsne229a1e7c05c",
        "x-rapidapi-host": "twitter154.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    data = data['results']
    print("number_of_tweets")
    print(len(data))
    clean_text_list = []
    for ind, tweet in enumerate(data):
        clean_text_list.append("Tweet " + str(ind + 1) + ": " + tweet['text'].replace("\n", " "))

    single_str_prompt = "\n".join(clean_text_list)
    return single_str_prompt


# Function to extract article text from Seeking Alpha
def article_text_extraction(id_list):
    """
    Extracts the full text of articles from Seeking Alpha given a list of article IDs.

    :param id_list: List of article IDs
    :return: List of article texts
    """

    url = "https://seeking-alpha.p.rapidapi.com/analysis/v2/get-details"

    headers = {
        "x-rapidapi-key": "799f09a0f2msh88cbc5a33cc1067p1d87e3jsne229a1e7c05c",
        "x-rapidapi-host": "seeking-alpha.p.rapidapi.com"
    }

    article_text_list = []
    for id in id_list:
        querystring = {"id": id}
        response = requests.get(url, headers=headers, params=querystring)
        print(response)
        data = response.json()['data']['attributes']['content']
        soup = BeautifulSoup(data)
        article_text_list.append(soup.get_text().strip())
    return article_text_list


# Function to save new article data
def save_new_article_data(stock_ticker):
    """
    Fetches new article data and saves it to a file.
    """

    id_list = get_seeking_alpha_article_ids()
    article_text_list = article_text_extraction(id_list)

    with open(stock_ticker + '_article_data.joblib', 'wb') as f:  # with statement avoids file leak
        joblib.dump(article_text_list, f)


# Function to load saved article data
def load_article_data(stock_ticker):
    """
    Loads previously saved article data from a file.

    :return: List of article texts
    """
    with open(stock_ticker + '_article_data.joblib', 'rb') as f:  # with statement avoids file leak
        article_text_list = joblib.load(f)
    return article_text_list


# Function to analyze Twitter sentiment
def twitter_sentiment_breakdown(single_str_prompt, ticker='sofi'):
    """
    Analyzes sentiment of tweets related to a given stock ticker.

    :param single_str_prompt: String containing formatted tweets
    :param ticker: Stock ticker symbol (default: 'sofi')
    """

    template = (
            "This is a collection of tweets of " + ticker + " stock:"
                                                            "\n"
                                                            "'{context_str}'"
                                                            "\n"
                                                            "As a world class financial analyst, analyze these tweets to extract sentiment."
                                                            "A tweet can represent positive, negative or neutral sentiment."
                                                            "For each tweet break it down. "

    )

    qa_template = PromptTemplate(template)
    prompt = qa_template.format(context_str=single_str_prompt)

    classify = dspy.ChainOfThought('question -> answer', n=1)
    # 2) Call with input argument.
    response = classify(question=prompt)
    print(response.answer)


# Function to sanitize text using LLM
def sanitize_text_for_yaml_using_LLM(text, llm):
    """
    Sanitizes text using a language model to ensure it is clean.

    :param text: Input text to sanitize
    :param llm: Language model to use for sanitization
    :return: Sanitized text
    """
    template_2 = (
        "This is a financial analysis text:"
        "\n"
        "{context_str}"
        "\n"
        "The first few words might not belong to the original text. Remove them and return the clean text."
        "The result must be a coherent text."
        "Remove the first few word that do not belong to the overall text."

    )

    qa_template = PromptTemplate(template_2)
    prompt = qa_template.format(context_str=text)
    out = llm.complete(prompt)
    return out.text


# Function to extract price target from text
def extract_price_target(text, llm):
    """
    Extracts the price target from a given financial analysis text.

    :param text: Financial analysis text
    :param llm: Language model to use for extraction
    :return: Extracted price target or -1 if not found
    """

    template = (
        "This is a financial analysis text:"
        "\n"
        "{context_str}"
        "\n"
        "Extract the price target if it exists in the text."
        "Just return the price target, if it does not exist, return -1"

    )
    qa_template = PromptTemplate(template)
    prompt = qa_template.format(context_str=text)
    classify = dspy.Predict('text -> result', n=1)
    response = classify(text=prompt)
    return response.result


# Function to analyze Seeking Alpha articles
def seeking_alpha_article_breakdown(single_article_text, ticker='AMD'):
    """
    Analyzes a Seeking Alpha article to determine sentiment and extract price target.

    :param single_article_text: Text of the Seeking Alpha article
    :param llm: Language model to use for analysis
    :param ticker: Stock ticker symbol (default: 'AMD')
    :return: Analysis result
    """
    template = (
            "This is a financial analysis of the stock " + ticker + ":"
                                                                    "\n"
                                                                    "{context_str}"
                                                                    "\n"
                                                                    "As a world class financial analyst, analyze this text to determine whether the author is bullish, bearish or neutral."
                                                                    "Extract the price target if it exists in the text."

    )

    qa_template = PromptTemplate(template)
    prompt = qa_template.format(context_str=single_article_text)

    classify = dspy.ChainOfThought('question -> answer', n=1)
    response = classify(question=prompt)
    return response.answer


# Function to set up and run RAG pipeline
def setup_RAG_pipeline(out_clean_article_list):
    """
    Sets up and runs a Retrieval-Augmented Generation (RAG) pipeline for financial analysis.

    :param out_clean_article_list: List of cleaned article texts
    :return: Query engine for RAG
    """
    text_list = out_clean_article_list
    documents = [Document(text=t) for t in text_list]

    parser = SentenceSplitter(chunk_size=450, chunk_overlap=100)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/llm-embedder", device='cuda')

    Settings.embed_model = embed_model
    Settings.llm = llm_gemini

    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)

    reranker = FlagEmbeddingReranker(
        top_n=7,
        model="BAAI/bge-reranker-large",
    )

    raw_query_engine = index.as_query_engine(
        similarity_top_k=15, node_postprocessors=[reranker], verbose=True)

    # hyde = HyDEQueryTransform(include_original=True)
    # raw_query_engine = TransformQueryEngine(raw_query_engine, query_transform=hyde)

    res = raw_query_engine.query("What are some bullish arguments mentioned in the articles?")
    print(res.response)
    print("----------------")
    res = raw_query_engine.query("What are some bearish arguments mentioned in the articles?")
    print(res.response)
    print("----------------")
    res = raw_query_engine.query("What are some bullish catalysts mentioned in the articles?")
    print(res.response)
    print("----------------")
    res = raw_query_engine.query(
        "As a financial analyst, how would you interpret the overall sentiment in the articles?")
    print(res.response)
    print("----------------")
    return raw_query_engine


def define_stock_analyzer_agent(raw_query_engine):
    """
    Defines and initializes a stock analyzer agent using the RAG query engine.

    :param raw_query_engine: Query engine from the RAG pipeline
    """
    tool = QueryEngineTool.from_defaults(
        raw_query_engine, name="rag_tool", description="Provides recent financial analysis about AMD."
                                                       "Use a detailed plain text question as input to the tool.",
        return_direct=True
    )
    agent = ReActAgent.from_tools([tool], llm=llm_gemini, verbose=True)

    res = agent.chat("What are some bullish arguments mentioned in the articles?")
    print(res.response)
    print("----------------")
    res = agent.chat("What are some bearish arguments mentioned in the articles?")
    print(res.response)
    print("----------------")
    res = agent.chat("What are some bullish catalysts mentioned in the articles?")
    print(res.response)
    print("----------------")


# Function to initialize Gemini LLM
def get_llamaindex_gemini():
    llm_gemini = Gemini(model_name="models/gemini-1.0-pro", api_key=gemini_api_key, safety_settings=SAFE,
                        temperature=0.01,
                        max_tokens=10000)
    return llm_gemini


# Main execution block
if __name__ == "__main__":
    # Flag to determine whether to use saved file or fetch new data
    use_saved_file = True
    get_new_article_data = False
    # save_new_article_data()
    stock_ticker = 'acmr'
    # Initialize Gemini model for natural language processing tasks
    gemini = dspy.Google(model="models/gemini-1.0-pro", api_key=gemini_api_key, safety_settings=SAFE,
                         max_output_tokens=10000)
    dspy.configure(lm=gemini)

    # Analyze Twitter sentiment
    print("Top Tweet Sentiment Breakdown")
    single_str_prompt = twitter_data_extractor(stock_ticker=stock_ticker)  # Extract tweets about the stock
    twitter_sentiment_breakdown(single_str_prompt, ticker=stock_ticker)  # Analyze sentiment of tweets
    print("-----------------")

    # Load article data from file
    if get_new_article_data:
        save_new_article_data(stock_ticker)

    article_data = load_article_data(stock_ticker)

    # Initialize Gemini LLM for article analysis
    llm_gemini = get_llamaindex_gemini()

    # Pause to avoid hitting API rate limits
    time.sleep(20)

    # Use parallel processing for faster execution
    with Parallel(n_jobs=-1, backend='threading') as parallel:
        if use_saved_file:
            # Clean and sanitize article data using LLM
            out_clean_article_list = parallel(
                (delayed(sanitize_text_for_yaml_using_LLM)(set, llm_gemini) for set in article_data))

            with open(stock_ticker + '_article_data_clean.joblib', 'wb') as f:
                joblib.dump(out_clean_article_list, f)

        else:
            # Load pre-cleaned article data from file
            with open(stock_ticker +'_article_data_clean.joblib', 'rb') as f:
                out_clean_article_list = joblib.load(f)

        time.sleep(40)
        # Analyze articles in parallel
        article_break_down_result = parallel(
            (delayed(seeking_alpha_article_breakdown)(single_article_text, stock_ticker) for single_article_text in
             out_clean_article_list))

        # Pause to avoid hitting API rate limits
        time.sleep(40)

        # Extract price targets from articles in parallel
        price_target_list = parallel((delayed(extract_price_target)(set, llm_gemini) for set in out_clean_article_list))

    # Filter and process price targets
    filtered = []
    for res in price_target_list:
        if float(res) > -1:
            filtered.append(float(res))

    # Calculate and display mean price target
    print("Seeking Alpha Mean Price Target")
    print(sum(filtered) / len(filtered))
    print("-----------------")

    # Display article analysis results
    print("Seeking Alpha Article Analysis")
    for ind, res in enumerate(article_break_down_result):
        print("Article " + str(ind + 1) + ":")
        print(res + "\n")

    # Run RAG (Retrieval-Augmented Generation) pipeline
    print("Seeking Alpha RAG")
    print("-----------------")
    setup_RAG_pipeline(out_clean_article_list)
