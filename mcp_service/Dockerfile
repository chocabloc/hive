FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir \
    asyncpg \
    mcp-server \
    langchain-google-genai \
    google-generativeai\
    neo4j==5.28.0

EXPOSE 8080
CMD ["python", "mcp_database_server.py"]
