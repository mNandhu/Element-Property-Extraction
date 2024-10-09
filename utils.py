"""
This script is used to get chunks of processed documents.
It uses the following steps:
1. Convert PDFs to .md files using LlamaParse
2. Split the .md files into chunks

The script uses the following environment variables:
1. LLAMA_CLOUD_API_KEY: The API key for the LlamaIndex API (For LlamaParse, If not found, PyMuPDF is used)

The script uses the following files:
1. data/pdfs: The directory containing the PDF files
2. data/markdowns: The directory containing the Markdown files
3. data/texts: The directory containing the text files (Optional)
"""

import os  # File Handling
import shutil  # Deleting Folders
from pathlib import Path  # Path Handling

from dotenv import load_dotenv  # Load Environment Variables
from langchain.schema import Document  # Document Schema
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split Document into Chunks
from langchain_community.document_loaders import DirectoryLoader  # Load md files

load_dotenv()

PDF_PATH = "data/pdfs"
MARKDOWN_PATH = 'data/markdowns/'
TEXT_PATH = "data/texts"


def getChunks():
    """
    Main Function to Get all chunks
    """
    reset = False  # Set to True if you want to delete the existing md files

    llamaParse_pdf2md(reset=reset)  # Summarize Paper and convert to md

    documents = load_md()
    chunks = split_text(documents).extend(getTextChunks())  # Add the text chunks

    return chunks


def llamaParse_pdf2md(reset=False, pdf_path=PDF_PATH, markdown_path=MARKDOWN_PATH):
    """
    Use LlamaIndex's LlamaParse to summarize PDFs
    Input PDFs from PDF_PATH
    Output as .md in MARKDOWN_PATH
    Only converts missing .md files
    :param pdf_path: Path to PDFs   (Default: data/pdfs)
    :param markdown_path: Path to store .md files (Default: data/markdowns)
    :param reset: False by Default, If True then delete all files in MARKDOWN_PATH
    """

    from llama_parse import LlamaParse

    with open("parsing_instruction.txt", "r") as f:
        parsing_instruction = f.read()

    pdf_search = Path(pdf_path).glob("*.pdf")
    parser = LlamaParse(result_type="markdown", parsing_instruction=parsing_instruction)

    if reset:  # If folder exists, delete and create new
        shutil.rmtree(markdown_path)
    if not os.path.exists(markdown_path):
        os.mkdir(markdown_path)

    for file in pdf_search:
        if not os.path.exists(markdown_path + file.name.replace('.pdf', '.md')):
            # if input(f"Process the file {file.name} (Y/N)? ").lower() != "y": continue
            with open(markdown_path + file.name.replace('.pdf', '.md'), 'w', encoding="utf-8") as f:
                docs = parser.load_data(pdf_path + '/' + file.name)
                for doc in docs:
                    f.write(doc.text)
    else:
        print("All PDFs are converted to .md successfully!")


def getTextChunks(text_path=TEXT_PATH) -> list[Document]:
    """
    Load text files and split them into chunks
    :param text_path: Path to text files
    :return: List of Documents
    """
    loader = DirectoryLoader(text_path, glob="*.txt")
    documents = loader.load()

    chunks = split_text(documents, chunk_size=500, chunk_overlap=10)
    return chunks


def load_md(markdown_path=MARKDOWN_PATH) -> list[Document]:
    """
    Load the .md files from the MARKDOWN_PATH
    :param markdown_path: Path to the .md files
    :return: List of documents
    """

    loader = DirectoryLoader(markdown_path, glob="*.md")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {markdown_path}")
    return documents


def split_text(documents: list[Document], chunk_size: int = 3500, chunk_overlap: int = 1000) -> list[Document]:
    """
    Split the documents into chunks.
    :param documents: List of documents to split
    :param chunk_size: Size of each chunk
    :param chunk_overlap: Overlap between chunks
    :return: List of chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                   length_function=len, add_start_index=True, keep_separator=False,
                                                   strip_whitespace=True, )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


if __name__ == "__main__":
    getChunks()
