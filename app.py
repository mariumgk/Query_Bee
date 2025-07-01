from flask import Flask, request, render_template, redirect, jsonify, session
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain LLM chain tools
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load API key from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file.")

app = Flask(__name__)
app.secret_key = "admin"
DOC_FOLDER = 'docs'
app.config['DOC_FOLDER'] = DOC_FOLDER

# Globals
vectorstore = None
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
memory = None  # LangChain memory object

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    files = os.listdir(app.config['DOC_FOLDER'])
    if request.method == 'POST':
        filename = request.form['filename']
        filepath = os.path.join(app.config['DOC_FOLDER'], filename)

        global vectorstore
        documents = load_file(filepath)
        chunks = split_chunks(documents)
        vectorstore = FAISS.from_documents(chunks, embedding_model)

        return redirect(f"/chat?filename={filename}")
    return render_template('upload.html', files=files)

@app.route('/chat', methods=['GET'])
def chat_page():
    filename = request.args.get("filename", "")
    history = session.get('chat_history', [])
    return render_template("chat.html", filename=filename, history=history)

@app.route('/chat', methods=['POST'])
def chat_ajax():
    try:
        data = request.get_json()
        user_question = data.get("message", "")
        if not user_question:
            return jsonify({"reply": "⚠️ No question received.", "sources": []})

        result = ask_question(user_question)

        chat_entry = {
            "user": user_question,
            "bot": result["answer"],
            "sources": result["sources"]
        }

        session.setdefault("chat_history", []).append(chat_entry)

        return jsonify({
            "reply": result["answer"],
            "sources": result["sources"]
        })

    except Exception as e:
        print("❌ Error in chat_ajax:", e)
        return jsonify({"reply": f"❌ Internal server error: {str(e)}", "sources": []}), 500

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    session['chat_history'] = []
    global memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return jsonify({"success": True})

def load_file(path):
    if path.endswith('.pdf'):
        loader = PyPDFLoader(path)
    elif path.endswith('.txt'):
        loader = TextLoader(path)
    elif path.endswith('.docx'):
        loader = Docx2txtLoader(path)
    else:
        raise ValueError("Unsupported file format")
    return loader.load()

def split_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def ask_question(query):
    global vectorstore, memory

    if vectorstore is None:
        return {"answer": "⚠️ No document loaded.", "sources": []}

    try:
        if memory is None:
           memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="result" 
)

        from langchain.chains import ConversationalRetrievalChain
        from langchain.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI

        # Prompt template
        template = """You are QueryBee, a helpful assistant that answers questions based on the provided context.

Context:
{context}

Chat history:
{chat_history}

Question: {question}
Answer:"""

        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=template
        )

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            output_key="result"  # ✅ THIS fixes the memory error!
        )
    
        response = chain.invoke({"question": query})  # ✅ use invoke not __call__
        docs = response.get("source_documents", [])
    
        sources = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown file")
            page = doc.metadata.get("page", "Unknown page")
            sources.append(f"📄 Chunk {i+1}: Page {page} of {source}")
    
        return {"answer": response["result"], "sources": sources}
    
    except Exception as e:
        print("❌ ask_question error:", e)
        return {"answer": f"❌ Error: {str(e)}", "sources": []}

if __name__ == '__main__':
    os.makedirs("docs", exist_ok=True)
    app.run(debug=True)
# To run this app, ensure you have Flask and LangChain installed