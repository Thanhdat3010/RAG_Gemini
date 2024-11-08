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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ChemGenieBot:
    def __init__(self, api_key, folder_path):
        logging.info("Đang khởi tạo ChemGenieBot...")
        self.api_key = api_key
        self.folder_path = folder_path
        self.embedding_cache_path = os.path.join(folder_path, "embedding_cache.pkl")
        self.files_hash_path = os.path.join(folder_path, "files_hash.pkl")
        self.setup_model()
        self.load_and_process_documents()
        self.setup_qa_chain()
        self.conversation_history = []

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

    def get_files_hash(self, file_paths):
        """Tạo hash để theo dõi thay đổi của các file"""
        hash_dict = {}
        for file_path in file_paths:
            try:
                mtime = os.path.getmtime(file_path)
                size = os.path.getsize(file_path)
                file_hash = f"{file_path}_{mtime}_{size}"
                hash_dict[file_path] = hashlib.md5(file_hash.encode()).hexdigest()
            except Exception:
                continue
        return hash_dict

    def load_and_process_documents(self):
        logging.info("Bắt đầu quá trình đọc tài liệu...")
        # Tìm tất cả các file
        pdf_files = glob.glob(os.path.join(self.folder_path, "*.pdf"))
        word_files = glob.glob(os.path.join(self.folder_path, "*.docx*"))
        txt_files = glob.glob(os.path.join(self.folder_path, "*.txt"))
        all_files = pdf_files + word_files + txt_files

        if not all_files:
            raise ValueError("No supported documents found in the specified folder.")

        # Tính hash của các file hiện tại
        current_files_hash = self.get_files_hash(all_files)
        
        # Kiểm tra cache và hash cũ
        should_update = True
        if os.path.exists(self.embedding_cache_path) and os.path.exists(self.files_hash_path):
            with open(self.files_hash_path, 'rb') as f:
                old_files_hash = pickle.load(f)
                
            # So sánh hash cũ và mới
            if current_files_hash == old_files_hash:
                logging.info("No changes detected in documents. Loading from cache...")
                with open(self.embedding_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    texts = cached_data['texts']
                    should_update = False

        if should_update:
            logging.info("Phát hiện thay đổi hoặc chạy lần đầu. Đang xử lý tài liệu...")
            texts = []
            
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
                    logging.info(f"Successfully processed {file_path}")
                    
                except Exception as e:
                    logging.warning(f"Failed to process {file_path}: {str(e)}")
                    continue
            
            if not texts:
                raise ValueError("No documents were successfully processed.")

            # Lưu texts và hash mới
            cache_data = {
                'texts': texts
            }
            with open(self.embedding_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            with open(self.files_hash_path, 'wb') as f:
                pickle.dump(current_files_hash, f)
            logging.info("Saved new texts and file hashes to cache")

        # Tạo embeddings và vector store
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
        template = """Use the following pieces of context to answer the question at the end. 
        If the question is about casual conversation, feel free to chat with the user. 
        If the question is related to knowledge, answer that you don't know if you're unsure. 
        Keep the answer as concise as possible. 
        Always answer in Vietnamese Language.
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
        self.conversation_history.append(f"User: {question}")
        
        context_with_history = "\n".join(self.conversation_history)
        result = self.qa_chain({"query": context_with_history})
        
        answer = result["result"]
        self.conversation_history.append(answer)
        
        logging.info("Đã tạo câu trả lời")
        return answer

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
            return jsonify({'error': 'No message provided'}), 400
        
        response = bot.ask_question(message)
        return jsonify({
            'response': response
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    logging.info("Đang khởi động ứng dụng ChemGenie Bot...")
    warnings.filterwarnings("ignore")
    app.run(debug=True) 