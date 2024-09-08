import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from redundant_filter_retriever import RedundantFilterRetriever

CHROMA_DIR = 'facts-chat-chroma'

load_dotenv()

chat_llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key=os.getenv('GROQ_API_KEY'),
    verbose=True
)

huggin_face_embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-l6-v2'
)

chroma_db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=huggin_face_embeddings
)

# retriever = chroma_db.as_retriever() # using the default retriever

retriever = RedundantFilterRetriever(embeddings=huggin_face_embeddings, db=chroma_db) # using the custom retriever

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chat_llm
    | StrOutputParser()
)

# res = retrieval_chain.invoke("Tell me one interesting fact about English language.")
# default retriever | ans - One interesting fact about the English language is that "Dreamt" is the only English word that ends with the letters "mt."
# custom retriever | ans - One interesting fact about the English language is that "Dreamt" is the only English word that ends with the letters "mt."

# res = retrieval_chain.invoke("What is the longest english word?")
# default retriever | ans - According to the provided context, the longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'
# custom retriever | ans - According to the provided context, the longest word in the English language is 'pneumonoultramicroscopicsilicovolcanoconiosis.'

while True:
    question = input("Ask a question: >> ")
    res = retrieval_chain.invoke(question)
    print(res)
