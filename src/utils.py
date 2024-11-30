import io
import re

import docx
import fitz
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap

from src.config import Config


def llm_chat_chain(query: str, context: str, history: str):
    template = """Answer the question based only on the following context and history:
    Context : {context}
    History : {history}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    chain = RunnableMap({
        "context": lambda x: context,
        "question": lambda x: x["question"],
        "history": lambda x: history
    }) | prompt | Config.chat_client | output_parser
    response = chain.invoke({"question": query})
    return response


# Function to extract text from a PDF file
def extract_pdf_text(pdf_file: bytes) -> str:
    try:
        # Open the PDF file from the byte stream
        with fitz.open(stream=pdf_file, filetype="pdf") as pdf_document:
            text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text += page.get_text("text")
            return text
    except Exception as e:
        raise ValueError(f"Error processing PDF file: {e}")


# Function to extract text from a DOCX file
def extract_docx_text(docx_file: bytes) -> str:
    doc = docx.Document(io.BytesIO(docx_file))
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


# Function to extract text from a TXT file
def extract_txt_text(txt_file: bytes) -> str:
    text = txt_file.decode("utf-8")
    return text


# Function to split long text into chunks for better embedding
def chunk_text(text: str, chunk_size: int = 3500) -> list:
    # Split text into smaller chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks


# Preprocess the text data by removing unwanted characters and tokenizing
def preprocess_text(text: str) -> str:
    # Remove extra newlines, tabs, and spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    # Further preprocessing, like removing special characters, stopwords, etc.
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text
