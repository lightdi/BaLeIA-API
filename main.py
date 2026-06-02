from fastapi import FastAPI, File, UploadFile, Body
import requests
import base64
import uvicorn
import os

app = FastAPI(root_path=os.environ.get("ROOT_PATH", ""))

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://ollama:11434/api/generate")



@app.post("/ocr")
async def ocr_imagem(
    image_base64: str = Body(''),
    key: str = Body(''),
    model: str = Body(''),
    system_message: str = Body('')
):
    
    if key != "b1a919be-94a2-40c9-983a-f6b7893f7800":
        return {
            "error": "Chave inválida!"
        }

    if system_message: 
        system_mensage = system_message
    else:
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
        Retorne apenas o texto puro, sem nenhum texto adicional.
    """

    if model:
        model = model
    else:
        model = "glm-ocr:latest"

    payload = {
        "model": model,
        "prompt": system_mensage,
        "images": [image_base64],
        "stream": False,        
        "options": {
            "options": {
                "num_ctx": 512,
                "num_predict": 400,
                "num_gpu": 999
            }
        }   
    }

    response = requests.post(OLLAMA_API_URL, json=payload)
    result = response.json()
    return {
        "text": result.get("response", ""),
        "model": result.get("model"),
        "done": result.get("done"),
        "done_reason": result.get("done_reason"),
        "tempo_total": round(
            result.get("total_duration", 0) / 1e9,
            2
        ),
        "tokens_saida": result.get("eval_count", 0)
    }


@app.post("/llm")
async def llm_chat(
    user_message: str = Body(''),
    key: str = Body(''),
    model: str = Body(''),
    system_message: str = Body('')
):
    if key != "b1a919be-94a2-40c9-983a-f6b7893f7800":
        return {
            "error": "Chave inválida!"
        }

    if model:
        model = model
    else:
        model = "phi4-mini:latest"

    payload = {
        "model": model,
        "prompt": user_message,
        "stream": False,        
        "options": {
            "options": {
                "num_ctx": 2048,
                "num_predict": 1024,
                "num_gpu": 999
            }
        }   
    }

    if system_message:
        payload["system"] = system_message

    response = requests.post(OLLAMA_API_URL, json=payload)
    result = response.json()
    return {
        "text": result.get("response", ""),
        "model": result.get("model"),
        "done": result.get("done"),
        "done_reason": result.get("done_reason"),
        "tempo_total": round(
            result.get("total_duration", 0) / 1e9,
            2
        ),
        "tokens_saida": result.get("eval_count", 0)
    }


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )