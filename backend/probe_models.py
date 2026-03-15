import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
API_KEY = os.getenv("OLLAMA_API_KEY")

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
data = {"model": "gpt-oss:120b-cloud", "messages": [{"role": "user", "content": "hello"}]}

print("Testing /v1/chat/completions...")
r = requests.post(f"{OLLAMA_BASE_URL}/v1/chat/completions", headers=headers, json=data)
print(r.status_code)
print(r.text)
