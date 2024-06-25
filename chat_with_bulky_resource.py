import streamlit as st
import os
import time
import requests
from PyPDF2 import PdfReader
from docx import Document
from streamlit_float import *


float_init(theme=True, include_unstable_primary=False)
# Assuming you have a function to process chat input and generate responses
def process_chat_input(input_text):
    # This function should call your chat model and return the response
    # Dummy function for illustration
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

def chat_content():
    user_input = st.session_state.content
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Process the input to get a response from the Flask API
        start_time = time.time()
        response = process_chat_input(user_input)
        end_time = time.time()
        t = end_time - start_time
        
        st.session_state.messages.append({"role": "assistant", "content": response['answer'], "sources": response['sources']})

# Initialize session state for messages if not already present


def main():
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    root_directory = 'E:/RAG-last/Data'
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
        with st.container(border=True):
            with st.container():
                st.chat_input("Type your message...", key='content', on_submit=chat_content) 
                button_b_pos = "0rem"
                button_css = float_css_helper(width="2.2rem", bottom=button_b_pos, transition=0)
                float_parent(css=button_css)
            
            # Display chat messages
            for message in st.session_state['messages']:
                with st.chat_message(message['role']):
                    st.write(message['content'])
                    if 'sources' in message:
                        for i, source in enumerate(message['sources'], 1):
                            st.write(f"Source {i}: {source['source']}")
                    if 'time_taken' in message:
                        st.write("Time taken to process:", message['time_taken'])

            # File viewing functionality
            if st.session_state.get('view_file', False):
                st.header("File Viewing")
                st.write("## Document Content")
                st.write(st.session_state.get('file_content', ''), unsafe_allow_html=True)
                
                file_path = st.session_state.get('file_path', '')
                if file_path.endswith('.pdf'):
                    page_input = st.number_input("Enter page number (starts at 0):", min_value=0, format="%d", key="page_num")
                    if st.button("Go to Page"):
                        st.session_state.file_content = read_pdf(file_path, page_input)
                        st.write(st.session_state.file_content, unsafe_allow_html=True)
                
                if st.button("Close File"):
                    st.session_state.view_file = False
                    st.session_state.file_content = ""
                    st.session_state.file_path = ""



    with main_col:
        st.subheader("File Explorer")
        if current_path != root_directory and st.button("Go Up"):
            st.session_state['current_path'] = os.path.dirname(current_path)
        
        for item in files_and_dirs:
            item_path = os.path.join(current_path, item)
            if os.path.isdir(item_path):
                if st.button(f"📁 {item}", key=f"dir_{item}"):
                    st.session_state['current_path'] = item_path
            else:
                if st.button(f"📄 {item}", key=f"file_{item}"):
                    st.session_state.file_content = display_file_content(item_path)
                    st.session_state.view_file = True
                    st.session_state.file_path = item_path

        
    

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
        return str(e)

if __name__ == "__main__":
    main()
