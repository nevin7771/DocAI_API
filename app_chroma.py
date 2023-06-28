from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain,RetrievalQA,VectorDBQA
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
import os

load_dotenv('.env')

app = Flask(__name__)

chat_history = []

@app.route('/upload', methods=['POST'])
def upload_document():
    file = request.files['file']
    if file and file.filename.endswith('.pdf'):
        documents = []

        for file in os.listdir("docs"):
            if file.endswith(".pdf"):
                pdf_path = "./docs/" + file
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                doc_path = "./docs/" + file
                loader = Docx2txtLoader(doc_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                text_path = "./docs/" + file
                loader = TextLoader(text_path)
                documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(documents)

        global vectordb

            
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(documents, embeddings,persist_directory="./chroma_db")
        vectordb.persist()
        #pinecone_index.upsert(items=vectors)

        return jsonify({"message": "Document uploaded and processed successfully."})
    else:
        return jsonify({"message": "Invalid file format. Only PDF files are supported."}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"message": "Question is required."}), 400

    global chat_history
    # Fetch vector database from Pinecone
    # Check chat history for existing question/answer
    for entry in chat_history:
        if entry["question"] == question:
            answer = entry["answer"]
            return jsonify({"answer": answer})
        
    embeddings = OpenAIEmbeddings()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    vectordb=Chroma(embedding_function=embeddings,persist_directory="./chroma_db")
    #docs = vectordb.similarity_search(question)
    retriever = vectordb.as_retriever(search_type="mmr")


    #llm = ChatOpenAI()
    # Perform document retrieval based on the question
    documents = vectordb.similarity_search(question)

    if not documents:
        return jsonify({"answer": "I'm sorry, but there is no relevant information available for your question."})


    pdf_qa = ConversationalRetrievalChain.from_llm(
        OpenAI(),
        retriever=vectordb.as_retriever(search_type="mmr"),
        memory=memory
    )

    #qa = RetrievalQA.from_chain_type(
    #    llm=OpenAI(),
    #    chain_type="stuff",
    #    retriever=retriever
    #)
    #qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)


    answer = pdf_qa({"question": question})
    #answer = pdf_qa.run(question)
    chat_history.append({"question": question, "answer": answer})
    return jsonify({"answer": answer["answer"]})


if __name__ == '__main__':
    app.run(debug=True,port=8000,host="0.0.0.0")
