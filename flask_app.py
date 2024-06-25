import textwrap
import pickle
from langchain_community.llms import OpenAI, Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from flask import Flask, request, jsonify, send_from_directory, jsonify, abort
import os
import argparse
import requests
app = Flask(__name__)



class CustomLLM:
    def __init__(self, base_url):
        self.base_url = base_url

    def predict(self, question):
        response = requests.post(
            f"{self.base_url}/ask",
            json={'question': question}
        )
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")


def qa_system(query):
    DB_FAISS_PATH = 'E:/Doc_Ass_2/vector_xx'
    hugging_face_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-V2', model_kwargs={'device': 'cpu'})

    # Step 2: Load FAISS vector store with embeddings
    faiss_db = FAISS.load_local(DB_FAISS_PATH, embeddings=hugging_face_embeddings, allow_dangerous_deserialization=True)
    retriever = faiss_db.as_retriever(search_kwargs={"k": 5})
    # Prompt
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    args = parse_arguments()
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = Ollama(model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents= not args.hide_source
    )

    result = qa_chain(query)
    return result

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    return wrap_text_preserve_newlines(llm_response['result']), llm_response["source_documents"]#.metadata['source']
 


def parse_arguments():
    parser = argparse.ArgumentParser(description='Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()



@app.route('/ask', methods=['POST'])
def ask():
    content = request.json
    question = content.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    llm_response = qa_system(question)
    answer = wrap_text_preserve_newlines(llm_response['result'])
    if answer.lower().startswith("i don't know"):
        sources= ''
    else:
        sources = [source.metadata for source in llm_response["source_documents"]]

    return jsonify({'answer':answer, 'sources':sources})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
