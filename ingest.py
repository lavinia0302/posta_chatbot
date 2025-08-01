from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import shutil
import os

# Șterge vechiul vectorstore (dacă există)
if os.path.exists("vectorstore"):
    shutil.rmtree("vectorstore")

# Încarcă fișierele .txt din folderul "posta_romana"
txt_loader = DirectoryLoader(
    "posta_romana",
    glob="**/*.txt",
    loader_cls=lambda path: TextLoader(path, encoding="utf-8")
)
documents = txt_loader.load()

print(f"Am încărcat {len(documents)} fișiere .txt")

# Verifică dacă documentele au conținut (debug)
for doc in documents:
    if not doc.page_content.strip():
        print(f"⚠️ Fisier gol: {doc.metadata['source']}")

# Creează embeddings
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

# Creează vectorstore
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("vectorstore")

print(f"✅ Vectorstore generat cu {len(documents)} documente din .txt.")
