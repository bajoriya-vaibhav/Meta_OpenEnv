FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build BM25 index at image build time (not runtime)
RUN python -c "from search.bm25_index import BM25Index; BM25Index.build_all()"

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
