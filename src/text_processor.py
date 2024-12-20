from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
import logging
import os
# File này để xử lý tài liệu
class DocumentProcessor:
    @staticmethod
    def load_document(file_path: str):
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            return PyPDFLoader(file_path)
        elif file_extension in ['.doc', '.docx']:
            return UnstructuredWordDocumentLoader(file_path)
        elif file_extension == '.txt':
            return TextLoader(file_path, encoding='utf-8')
        raise ValueError(f"Unsupported file type: {file_extension}")

    @staticmethod
    def process_document(file_path: str):
        try:
            loader = DocumentProcessor.load_document(file_path)
            pages = loader.load_and_split()
            context = "\n\n".join(str(p.page_content) for p in pages)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            texts = text_splitter.split_text(context)
            
            return texts
        except Exception as e:
            logging.error(f"Error processing document {file_path}: {str(e)}")
            raise