import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import warnings
import os
import glob
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import logging
import hashlib
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ChemGenieBot:
    def __init__(self, api_key, folder_path):
        logging.info("Đang khởi tạo ChemGenieBot...")
        self.api_key = api_key
        self.folder_path = folder_path
        self.processed_files = []
        self.setup_model()
        self.load_and_process_documents()
        self.setup_qa_chain()
        self.conversation_history = []
        self.max_context_messages = 3

    def setup_model(self):
        logging.info("Đang cài đặt model Gemini...")
        genai.configure(api_key=self.api_key)
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.2,
            convert_system_message_to_human=True
        )
        logging.info("Hoàn tất cài đặt model")

    def load_and_process_documents(self):
        logging.info("Bắt đầu quá trình đọc tài liệu...")
        # Tìm tất cả các file
        pdf_files = glob.glob(os.path.join(self.folder_path, "*.pdf"))
        word_files = glob.glob(os.path.join(self.folder_path, "*.docx*"))
        txt_files = glob.glob(os.path.join(self.folder_path, "*.txt"))
        all_files = pdf_files + word_files + txt_files

        if not all_files:
            raise ValueError("No supported documents found in the specified folder.")

        texts = []
        self.processed_files = []
        
        for file_path in all_files:
            logging.info(f"Đang xử lý file: {file_path}")
            try:
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.pdf':
                    loader = PyPDFLoader(file_path)
                elif file_extension in ['.doc', '.docx']:
                    loader = UnstructuredWordDocumentLoader(file_path)
                elif file_extension == '.txt':
                    loader = TextLoader(file_path, encoding='utf-8')
                
                pages = loader.load_and_split()
                context = "\n\n".join(str(p.page_content) for p in pages)
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=52)
                texts.extend(text_splitter.split_text(context))
                self.processed_files.append(file_path)
                logging.info(f"Successfully processed {file_path}")
                
            except Exception as e:
                logging.warning(f"Failed to process {file_path}: {str(e)}")
                continue
        
        if not texts:
            raise ValueError("No documents were successfully processed.")

        # Tạo embeddings và vector store trực tiếp
        logging.info("Đang tạo embeddings và vector store...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )
        
        logging.info("Đang xây dựng vector index...")
        self.vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})
        logging.info("Hoàn tất tạo vector index")

    def setup_qa_chain(self):
        logging.info("Đang thiết lập QA chain...")
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Always answer Vietnamese Language.
            {context}
            Question: {question}
            Helpful Answer:"""
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            self.model,
            retriever=self.vector_index,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        logging.info("Hoàn tất thiết lập QA chain")

    def ask_question(self, question):
        logging.info(f"Nhận được câu hỏi: {question}")
        try:
            # Sử dụng trực tiếp câu hỏi với RAG, không thêm context từ lịch sử
            result = self.qa_chain({"query": question})
            answer = result.get("result", "")
            
            if not answer or answer.strip() == "":
                answer = "Xin lỗi, tôi không thể tạo câu trả lời. Vui lòng thử lại."
            
            return answer
            
        except Exception as e:
            logging.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")
            return "Đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại."

    def get_processed_files(self):
        return self.processed_files

# Khởi tạo Flask app và chatbot
logging.info("Đang khởi tạo ứng dụng Flask...")
app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = "AIzaSyD3SQ-8fHNjPnEGn4gegLk57JNNQO8U8lI"
FOLDER_PATH = "data"
bot = ChemGenieBot(GOOGLE_API_KEY, FOLDER_PATH)

@app.route('/chat', methods=['POST'])
def chat():
    logging.info("Nhận được yêu cầu chat mới")
    try:
        message = request.form.get('message')
        if not message:
            return jsonify({'error': 'Không có tin nhắn được cung cấp'}), 400
        
        response = bot.ask_question(message)
        if not response or response.strip() == "":
            return jsonify({'error': 'Không thể tạo câu trả lời'}), 500
            
        return jsonify({
            'response': response
        })
    
    except Exception as e:
        logging.error(f"Lỗi server: {str(e)}")
        return jsonify({'error': 'Lỗi server nội bộ'}), 500

if __name__ == "__main__":
    logging.info("Đang khởi động ứng dụng ChemGenie Bot...")
    warnings.filterwarnings("ignore")
    app.run(debug=True) 