import os
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import chainlit as cl

# === Load and prepare data ===
with open("combined_data.json", "r") as f:
    raw_data = json.load(f)

all_docs = [
    Document(page_content=entry["content"], metadata=entry["metadata"])
    for entry in raw_data
]

# === Split documents into chunks ===
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
chunked_docs = splitter.split_documents(all_docs)

# === Use your fine-tuned Hugging Face embeddings ===
embedding_model = HuggingFaceEmbeddings(
    model_name="AneetaXavier/reformer-pilates-embed-ft-49fc1835-9968-433d-9c45-1538ea91dcc9"
)

# === Set up FAISS vector store ===
vectorstore = FAISS.from_documents(chunked_docs, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# === Load LLM ===
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# === Chainlit start event ===
@cl.on_chat_start
async def start():
    await cl.Message("🤸 Ready! Ask me anything about Reformer Pilates.").send()
    cl.user_session.set("qa_chain", qa_chain)

# === Chainlit message handler ===
@cl.on_message
async def handle_message(message: cl.Message):
    chain = cl.user_session.get("qa_chain")
    if chain:
        try:
            response = chain.run(message.content)
        except Exception as e:
            response = f"⚠️ Error: {str(e)}"
        await cl.Message(response).send()
