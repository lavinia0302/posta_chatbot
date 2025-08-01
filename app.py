from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import pandas as pd




import pandas as pd


# Încarcă variabilele din .env
load_dotenv()

# Initializează Flask
app = Flask(__name__)
CORS(app)  # 🔓 Permite cereri din React (alt port)

# Încarcă embeddings și vectorstore
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# Creează lanțul QA
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
def reformuleaza_intrebare(original_question):
    prompt = f"""
    Reformulează întrebarea următoare într-o formă mai clară și specifică, folosind termeni folosiți de obicei în documente Poșta Română.

    Întrebare: "{original_question}"
    Reformulare:
    """
    try:
        response = llm.predict(prompt)
        reformulated = response.strip()
        print(f"[INTREBARE] Original: {original_question}")
        print(f"[REFORMULAT] Reformulated: {reformulated}")
        return reformulated
    except Exception as e:
        print("Eroare la reformulare:", e)
        return original_question  # fallback

# Endpoint pentru întrebări
@app.route("/chat", methods=["POST"])  # 🔁 folosește /chat în loc de /ask
def chat():
    data = request.get_json()
    question = data.get("question", "")

    if not question.strip():
        return jsonify({"error": "Întrebarea este goală."}), 400

    reformulated = reformuleaza_intrebare(question)
    result = qa_chain({"query": reformulated})

    answer = result.get("result", "Nu am găsit un răspuns.")
    # Post-procesare simplă pentru răspunsuri mai aerisite
    answer = answer.replace("##", "\n\n🔹 ").replace(" - ", "\n• ")
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)