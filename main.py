from fastapi import FastAPI, File, UploadFile
import requests
import base64
import uvicorn

# Edição de Imagens 
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import io

#paddleocr
from paddleocr import PaddleOCR
import numpy as np

app = FastAPI()

OLLAMA_API_URL = "http://200.129.71.149:11434/api/generate"

# Inicializa uma vez
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="pt"
)
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    engine="paddle",
    lang="pt",
)

@app.post("/ocr")
async def ocf_imagem( file: UploadFile = File(...)):
    image_bytes = await file.read()

    img = Image.open(io.BytesIO(image_bytes))

    # resize
    max_width = 512

    if img.width > max_width:

        ratio = max_width / img.width

        img = img.resize(
            (
                max_width,
                int(img.height * ratio)
            ),
            Image.LANCZOS
        )

    # grayscale
    img = img.convert("L")

    # contraste
    img = ImageEnhance.Contrast(img).enhance(2)

    # nitidez
    img = ImageEnhance.Sharpness(img).enhance(2)

    # remover ruido
    img = img.filter(ImageFilter.MedianFilter())

    buffer = io.BytesIO()

    img.save(
        buffer,
        format="JPEG",
        quality=85,
        optimize=True
    )

    image_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")
    
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

    system_mensage2 = """
        Extraia TODO o texto da imagem.
        Mantenha parágrafos, acentos e pontuação.
        Não resuma.
        Retorne apenas o texto puro.
    """

    #model = "qwen2.5vl:7b-gpu"
    model = "glm-ocr:latest"
    #model = "llava:7b"
    #model = "openbmb/minicpm-v2.6"

    payload = {
        "model": model,
        "prompt": system_mensage2,
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

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image_np = np.array(image)

    result = ocr.ocr(image_np, cls=True)

    textos = []

    for bloco in result:

        if bloco is None:
            continue

        for linha in bloco:

            texto = linha[1][0]

            confianca = linha[1][1]

            textos.append({
                "texto": texto,
                "confianca": round(float(confianca), 4)
            })

    texto_final = "\n".join(
        [t["texto"] for t in textos]
    )

    return {
        "text": texto_final,
        "linhas": textos
    }

@app.post("/paddleocr")
async def paddle_ocr_image(
    file: UploadFile = File(...)
):

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image_np = np.array(image)

    result = ocr.predict(image_np)

    textos = []

    for page in result:

        if "rec_texts" in page:

            textos.extend(
                page["rec_texts"]
            )

    texto_final = "\n".join(textos)

    return {
        "text": texto_final
    }


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )