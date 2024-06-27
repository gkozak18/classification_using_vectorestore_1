import os
import configparser


config = configparser.ConfigParser()
config.read('config.ini')

os.environ["OPENAI_API_KEY"] = config.get('openai', 'OPENAI_API_KEY')