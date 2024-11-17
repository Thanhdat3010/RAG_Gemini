import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import logging
import glob
import os
from .text_processor import DocumentProcessor
from .document_store import DocumentStore
from .ranking import DocumentRanker
from .question_classifier import QuestionClassifier

class ChemGenieBot:
    def __init__(self, api_key, folder_path):
        logging.info("Đang khởi tạo ChemGenieBot...")
        self.api_key = api_key
        self.folder_path = folder_path
        self.setup_model()
        self.setup_components()
        self.load_and_process_documents()
        self.setup_qa_chain()

    def setup_model(self):
        genai.configure(api_key=self.api_key)
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.2,
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )

    def setup_components(self):
        self.doc_processor = DocumentProcessor()
        self.doc_store = DocumentStore(self.embeddings)
        self.ranker = DocumentRanker()
        self.question_classifier = QuestionClassifier(self.api_key)

    def load_and_process_documents(self):
        logging.info("Bắt đầu quá trình đọc tài liệu...")
        pdf_files = glob.glob(os.path.join(self.folder_path, "*.pdf"))
        word_files = glob.glob(os.path.join(self.folder_path, "*.docx"))
        txt_files = glob.glob(os.path.join(self.folder_path, "*.txt"))
        all_files = pdf_files + word_files + txt_files

        if not all_files:
            raise ValueError("Không tìm thấy tài liệu hỗ trợ trong thư mục.")

        for file_path in all_files:
            current_hash = self.doc_store.calculate_file_hash(file_path)
            if file_path not in self.doc_store.file_hashes or self.doc_store.file_hashes[file_path] != current_hash:
                logging.info(f"Đang xử lý file mới/đã thay đổi: {file_path}")
                try:
                    texts = self.doc_processor.process_document(file_path)
                    logging.info(f"Đã tạo được {len(texts)} chunks từ file {file_path}")
                    self.doc_store.update_vectors(file_path, texts)
                except Exception as e:
                    logging.error(f"❌ Lỗi khi xử lý {file_path}: {str(e)}")

        self.vector_index = self.doc_store.get_retriever()
        logging.info("Hoàn tất quá trình đọc và xử lý tài liệu")

    def setup_qa_chain(self):
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

    def ask_question(self, question):
        logging.info(f"Nhận được câu hỏi: {question}")
        try:
            # Phân loại câu hỏi
            if self.question_classifier.is_conversational(question):
                logging.info("Phát hiện câu hỏi giao tiếp, sử dụng LLM trực tiếp")
                return self.question_classifier.get_conversation_response(question)
            
            # Xử lý câu hỏi chuyên môn bằng RAG
            logging.info("Đang tìm kiếm tài liệu liên quan...")
            retrieved_docs = self.vector_index.get_relevant_documents(
                question, 
                fetch_k=10  # Giảm từ 20 xuống 10
            )
            logging.info(f"Tìm thấy {len(retrieved_docs)} tài liệu liên quan")
            
            logging.info("Bắt đầu rerank tài liệu...")
            reranked_docs = self.ranker.rerank_documents(question, retrieved_docs)
            logging.info("Hoàn thành rerank tài liệu")
            
            context = "\n\n".join(doc.page_content for doc in reranked_docs)
            
            logging.info("Đang tạo câu trả lời...")
            result = self.qa_chain.invoke({"query": question, "context": context})
            answer = result.get("result", "")
            
            if not answer or answer.strip() == "":
                logging.warning("Không thể tạo câu trả lời")
                return "Xin lỗi, tôi không thể tạo câu trả lời. Vui lòng thử lại."
            
            logging.info("Đã tạo xong câu trả lời")
            return answer
            
        except Exception as e:
            logging.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")
            return "Đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại."