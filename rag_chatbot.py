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
from pydantic import BaseModel, Field
from typing import List
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="Điểm đánh giá độ liên quan của tài liệu với câu hỏi.")

class ChemGenieBot:
    def __init__(self, api_key, folder_path):
        logging.info("Đang khởi tạo ChemGenieBot...")
        self.api_key = api_key
        self.folder_path = folder_path
        self.processed_files = []
        self.vector_store_path = "vector_store"
        self.file_hash_path = "processed_files_hash.pkl"
        self.file_hashes = self.load_file_hashes()
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

    def load_file_hashes(self):
        """Load existing file hashes from pickle file"""
        if os.path.exists(self.file_hash_path):
            with open(self.file_hash_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_file_hashes(self):
        """Save current file hashes to pickle file"""
        with open(self.file_hash_path, 'wb') as f:
            pickle.dump(self.file_hashes, f)

    def calculate_file_hash(self, file_path):
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def load_and_process_documents(self):
        logging.info("Bắt đầu quá trình đọc tài liệu...")
        pdf_files = glob.glob(os.path.join(self.folder_path, "*.pdf"))
        word_files = glob.glob(os.path.join(self.folder_path, "*.docx*"))
        txt_files = glob.glob(os.path.join(self.folder_path, "*.txt"))
        all_files = pdf_files + word_files + txt_files

        if not all_files:
            raise ValueError("No supported documents found in the specified folder.")

        # Khởi tạo embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )

        # Load hoặc tạo mới vector store
        if os.path.exists(self.vector_store_path):
            vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=embeddings
            )
        else:
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=self.vector_store_path
            )

        # Xác định các file đã bị xóa
        deleted_files = set(self.file_hashes.keys()) - set(all_files)
        if deleted_files:
            logging.info(f"Phát hiện {len(deleted_files)} file đã bị xóa")
            # Xóa các vector của file đã bị xóa
            for file_path in deleted_files:
                vector_store.delete(where={"source": file_path})
                del self.file_hashes[file_path]

        # Xử lý các file mới hoặc đã thay đổi
        for file_path in all_files:
            current_hash = self.calculate_file_hash(file_path)
            if file_path not in self.file_hashes or self.file_hashes[file_path] != current_hash:
                logging.info(f"Đang xử lý file mới/đã thay đổi: {file_path}")
                try:
                    # Xóa vector cũ nếu file đã tồn tại trước đó
                    if file_path in self.file_hashes:
                        logging.info(f"Xóa vector cũ của file: {file_path}")
                        vector_store.delete(where={"source": file_path})

                    # Xử lý và thêm vector mới
                    file_extension = os.path.splitext(file_path)[1].lower()
                    if file_extension == '.pdf':
                        loader = PyPDFLoader(file_path)
                    elif file_extension in ['.doc', '.docx']:
                        loader = UnstructuredWordDocumentLoader(file_path)
                    elif file_extension == '.txt':
                        loader = TextLoader(file_path, encoding='utf-8')
                    
                    logging.info(f"Đang load nội dung file: {file_path}")
                    pages = loader.load_and_split()
                    context = "\n\n".join(str(p.page_content) for p in pages)
                    
                    logging.info(f"Bắt đầu chunking file: {file_path}")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=52)
                    texts = text_splitter.split_text(context)
                    logging.info(f"Hoàn thành chunking file {file_path}: tạo được {len(texts)} chunks")
                    
                    # Thêm metadata source để tracking
                    metadatas = [{"source": file_path} for _ in texts]
                    logging.info(f"Bắt đầu tạo vector cho {len(texts)} chunks của file {file_path}")
                    vector_store.add_texts(texts, metadatas=metadatas)
                    logging.info(f"Hoàn thành tạo vector cho file: {file_path}")
                    
                    self.file_hashes[file_path] = current_hash
                    self.processed_files.append(file_path)
                    
                except Exception as e:
                    logging.error(f"❌ Lỗi khi xử lý {file_path}: {str(e)}")
                    continue

        vector_store.persist()
        self.vector_index = vector_store.as_retriever(search_kwargs={"k": 5})
        self.save_file_hashes()
        logging.info("Cập nhật vector store thành công")

    def setup_qa_chain(self):
        logging.info("Đang thiết lập QA chain...")
        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Always answer Vietnamese Language.
            {context}
            Question: {question}
            
            formatting rules:
            1. Ensure all chemical formulas use proper subscript notation (e.g., write CH₄ instead of CH4)
            2. Use → for reaction arrows (instead of ->)
            3. Use ⇌ for reversible reactions (instead of <->)
            4. Format your response with proper spacing and line breaks:
               - Use double line breaks between paragraphs
               - Use bullet points (•) for lists
               - Use bold (**) for important terms or concepts
               - Use proper indentation for chemical equations
            
            Helpful Answer:"""
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            self.model,
            retriever=self.vector_index,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        logging.info("Hoàn tất thiết lập QA chain")

    def rerank_documents(self, query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
        logging.info(f"Bắt đầu rerank {len(docs)} tài liệu...")
        prompt_template = PromptTemplate(
            input_variables=["query", "doc"],
            template="""On a scale of 1-10, rate the relevance of the following document to the query. Consider the specific context and intent of the query, not just keyword matches.
        Query: {query}
        Document: {doc}
        Relevance Score:"""
        )
        
        llm_chain = prompt_template | self.model.with_structured_output(RatingScore)
        
        scored_docs = []
        for idx, doc in enumerate(docs):
            input_data = {"query": query, "doc": doc.page_content}
            try:
                score = llm_chain.invoke(input_data).relevance_score
                scored_docs.append((doc, float(score)))
                # Log thông tin chi tiết về từng document
                logging.info(f"\nDocument {idx + 1}:")
                logging.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
                logging.info(f"Score: {score}")
                logging.info(f"Preview: {doc.page_content[:200]}...")  # Hiển thị 200 ký tự đầu
            except Exception as e:
                logging.error(f"Lỗi khi đánh giá document {idx + 1}: {str(e)}")
                scored_docs.append((doc, 0))

        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        # Log kết quả sau khi rerank
        logging.info("\nKết quả rerank:")
        for idx, (doc, score) in enumerate(reranked_docs[:top_n]):
            logging.info(f"\nTop {idx + 1}:")
            logging.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
            logging.info(f"Score: {score}")
            logging.info(f"Preview: {doc.page_content[:200]}...")

        return [doc for doc, _ in reranked_docs[:top_n]]

    def ask_question(self, question):
        logging.info(f"Nhận được câu hỏi: {question}")
        try:
            # Lấy tài liệu ban đầu
            retrieved_docs = self.vector_index.get_relevant_documents(question)
            
            # Rerank tài liệu
            reranked_docs = self.rerank_documents(question, retrieved_docs)
            
            # Tạo context từ các tài liệu đã rerank
            context = "\n\n".join(doc.page_content for doc in reranked_docs)
            
            # Lấy câu trả lời sử dụng context đã được rerank
            result = self.qa_chain({"query": question, "context": context})
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