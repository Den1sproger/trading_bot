import os

from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

API_KEY = os.getenv('api_key')
SECRET_KEY = os.getenv('secret_key')
API_TOKEN = os.getenv('TG_BOT_TOKEN')
GROUP_CHAT_ID = os.getenv('GROUP_CHAT_ID')