from fastapi import FastAPI, File, UploadFile
import requests
import base64
import uvicorn

app = FastAPI()

OLLAMA_API_URL = "http://200.129.71.149:11434/api/generate"


@app.post("/ocr")
async def ocf_imagem( file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    system_mensage = """
        Extraia exatamente o texto visível da página.
        Preserve:
        - títulos
        - quebras de linha
        - fórmulas
        - tabelas simples
        - numeração

        Ignore dedos, sombras e fundo.
        Não invente texto.
    """

    system_mensage2 = """
        Extraia TODO o texto da imagem.
        Mantenha parágrafos, acentos e pontuação.
        Não resuma.
        Retorne apenas o texto puro.
    """

    payload = {
        "model": "qwen2.5vl:7b",
        "messages": system_mensage2,
        "images": [image_base64],
        "stream": False,        
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    result = response.json()
    return {
        "text": result["response"]
    }
