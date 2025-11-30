import os

def get_api_key() -> str:
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    if API_KEY:
        return API_KEY
    else:
        try:
            with open('.env', 'r') as f:
                for line in f:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
            API_KEY = os.getenv("OPENROUTER_API_KEY")
            if API_KEY is None:
                raise ValueError("Please set the OPENROUTER_API_KEY environment variable (e.g. in .env)")
        except FileNotFoundError:
            raise ValueError("Please set the OPENROUTER_API_KEY environment variable (e.g. in .env)")
    return API_KEY

def get_base_url() -> str:
    BASE_URL = os.getenv("BASE_URL")
    if BASE_URL:
        return BASE_URL
    else:
        try:
            with open('.env', 'r') as f:
                for line in f:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
            BASE_URL = os.getenv("BASE_URL")
            if BASE_URL is None:
                raise ValueError("Please set the OPENROUTER_BASE_URL environment variable (e.g. in .env)")
        except FileNotFoundError:
            raise ValueError("Please set the OPENROUTER_BASE_URL environment variable (e.g. in .env)")
    return BASE_URL