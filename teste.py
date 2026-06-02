import requests
import base64

# URL da nossa API local rodando em FastAPI
url = "https://integramaker.ifpb.edu.br/ia-api/ocr"

# Envia a imagem local convertida em base64
with open("01.jpeg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

# Payload JSON aceito pelo endpoint ocr_imagem no main.py
payload = {
    "image_base64": image_base64,
    "key": "b1a919be-94a2-40c9-983a-f6b7893f7800",
    "model": "glm-ocr:latest",
    "system_message": "Extraia todo o texto da imagem e retorne apenas o texto puro."
}

# Realiza a requisição POST com os dados no corpo (JSON)
response = requests.post(url, json=payload, timeout=300)

if response.status_code == 200:
    res_data = response.json()
    if "error" in res_data:
        print("Erro da API:", res_data["error"])
    else:
        print("Texto extraído:")
        print(res_data.get("text"))
else:
    print(f"Erro na requisição HTTP {response.status_code}:")
    print(response.text)