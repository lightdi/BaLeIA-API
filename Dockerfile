# Imagem base oficial do Python (leve/slim)
FROM python:3.11-slim

# Define o diretório de trabalho no container
WORKDIR /app

# Configurações do Python para evitar gravação de arquivos .pyc e garantir logs imediatos
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copia e instala as dependências antes do código para aproveitar o cache do Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código do projeto para o container
COPY . .

# Explicita a porta em que a aplicação escuta
EXPOSE 8000

# Executa o uvicorn apontando para o app no main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
