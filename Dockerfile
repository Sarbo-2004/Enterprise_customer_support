FROM python:3.10

WORKDIR /app
 
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
 
COPY . .
 
# Build FAISS index during image build

RUN python ingest.py
 
EXPOSE 8080

EXPOSE 8501
 
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port 8080 & streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0"
 