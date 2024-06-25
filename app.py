import streamlit as st
import os
import requests
from PyPDF2 import PdfReader
from docx import Document

def process_chat_input(input_text):
    return f"Responding to: {input_text}"

def read_pdf(file_path, page_number=None):
    """Reads specified page of a PDF file."""
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        if page_number is not None and 0 <= page_number < len(reader.pages):
            return reader.pages[page_number].extract_text() or "No text found on this page."
        elif page_number is not None:
            return "Page number out of range."
        else:
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text() is not None])

def read_docx(file_path):
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def read_txt(file_path):
    with open(file_path, "r") as file:
        return file.read()

def app():
    root_directory = 'E:/Document_Assistant/Data'
    if 'current_path' not in st.session_state:
        st.session_state['current_path'] = root_directory
    if 'view_file' not in st.session_state:
        st.session_state.view_file = False
    if 'file_content' not in st.session_state:
        st.session_state.file_content = ""

    current_path = st.session_state['current_path']
    files_and_dirs = os.listdir(current_path)

    with st.sidebar:
        st.title("Chat History")          
    chat_col, main_col = st.columns([0.7, 0.3])

    with chat_col:
        st.subheader("Chat")
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []
        user_input = st.chat_input("Type your message...", key="chat_input")
        if user_input:
            st.session_state['messages'].append({"role": "user", "content": user_input})
            # Process the input to get a response from the Flask API
            response = process_chat_input(user_input)
            print('vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv',response)
            st.session_state['messages'].append({"role": "assistant", "content": response['answer'], "sources": response['sources']})

        for message in st.session_state['messages']:
            with st.chat_message(message['role']):
                st.write(message['content'])
                if 'sources' in message:
                    ii = 1
                    for source in message['sources']:
                        st.write(f"Source {ii}: {source['source']}, Page: {source['page']}")
                        ii += 1
            

        if st.session_state.view_file:
            st.header("File Viewing")
            page_input = st.number_input("Enter page number (starts at 0):", min_value=0, format="%d", key="page_num")
            if st.button("Go to Page"):
                file_type = st.session_state.file_path.split('.')[-1].lower()
                if file_type == 'pdf':
                    st.session_state.file_content = read_pdf(st.session_state.file_path, page_input)
                st.markdown("## Document Content")
                st.write(st.session_state.file_content, unsafe_allow_html=True)
            if st.button("Close File"):
                st.session_state.view_file = False
                st.session_state.file_content = ""



    with main_col:
        st.subheader("File Explorer")
        if current_path != root_directory and st.button("Go Up"):
            st.session_state['current_path'] = os.path.dirname(current_path)
        for item in files_and_dirs:
            item_path = os.path.join(current_path, item)
            if os.path.isdir(item_path):
                if st.button(f"📁 {item}", key=item):
                    st.session_state['current_path'] = item_path
            else:
                if st.button(f"📄 {item}", key=item):
                    st.session_state.file_content = display_file_content(item_path)
                    st.session_state.view_file = True

        
    

def display_file_content(file_path):
    file_type = file_path.split('.')[-1].lower()
    st.session_state.file_path = file_path
    if file_type == 'pdf':
        return read_pdf(file_path)
    elif file_type == 'docx':
        return read_docx(file_path)
    elif file_type == 'txt':
        return read_txt(file_path)
    return "Unsupported file type"

def process_chat_input(input_text):
    """Sends chat input to the Flask API and returns the response."""
    url = 'http://localhost:5001/ask'  # URL of the Flask API
    data = {'question': input_text}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return "Failed to get response from the server. Status code: {}".format(response.status_code)
    except requests.exceptions.RequestException as e:
        return str(e)  # Handle any exceptions that occur during the request


if __name__ == "__main__":
    app()
