from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
from sentence_transformers import SentenceTransformer

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

df = pd.read_csv("Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv", encoding="utf-8")

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

# Build proper Documents with metadata (no splitter needed)
docs = [
    Document(
        page_content=row["instruction"],
        metadata={
            "intent":    row["intent"],
            "category":  row["category"],
            "response":  row["response"],
            "flags":     row["flags"]
        }
    )
    for _, row in df.iterrows()
]

# Build FAISS directly — metadata stays attached
faiss_store = FAISS.from_documents(docs, hf_embeddings)
faiss_store.save_local("faiss_index")
