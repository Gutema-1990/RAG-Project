import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-V2', model_kwargs={'device': 'cpu'})
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    llm = Ollama(model="llama3", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), num_predict=-2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-V2', model_kwargs={'device': 'cpu'})
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]  # Return the response for display

def chat_content():
    user_input_x = st.session_state.content
    if user_input_x:
        st.session_state.messages.append({"role": "user", "content": user_input_x})
        response = user_input(user_input_x)
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    # CSS to fix the input box at the bottom and make the chat messages scrollable
    st.markdown(
        """
        <style>
        .chat-container {
            height: calc(100vh - 160px);
            overflow-y: auto;
            padding-bottom: 80px;
        }
        .chat-input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: white;
            padding: 10px;
            border-top: 1px solid #ddd;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.header("Chat with Your PDF")

    with st.container():
        chat_container = st.empty()

    with chat_container.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        st.markdown('</div>', unsafe_allow_html=True)

    with st.markdown('<div class="chat-input-container">', unsafe_allow_html=True):
        user_input_x = st.chat_input("Type your message...", key='content', on_submit=chat_content)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
