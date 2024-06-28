import os
import configparser
from langchain_openai import ChatOpenAI


config = configparser.ConfigParser()
config.read('config.ini')

os.environ["OPENAI_API_KEY"] = config.get('openai', 'OPENAI_API_KEY')

os.environ["LANGCHAIN_API_KEY"] = config.get('langsmith', 'LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = config.get('langsmith', 'LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_PROJECT"] =  config.get('langsmith', 'LANGCHAIN_PROJECT')
os.environ["LANGCHAIN_ENDPOINT"] = config.get('langsmith', 'LANGCHAIN_ENDPOINT')

llm_gpt4o = ChatOpenAI(temperature=0.8, model="gpt-4o", verbose=True)
llm_gpt35 = ChatOpenAI(temperature=0.7, model="gpt-4o", verbose=True)