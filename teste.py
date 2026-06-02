import requests
import base64

with open("01.jpeg", "rb") as f:
    img = base64.b64encode(f.read()).decode()

payload = {
    "model": "quen-gpu:latest",
    "prompt": "Extraia todo o texto da imagem e retorne apenas o texto.",
    "images": [img],
    "stream": False
}

r = requests.post(
    "http://200.129.71.149:11434/api/generate",
    json=payload,
    timeout=300
)

print(r.json()['response'])