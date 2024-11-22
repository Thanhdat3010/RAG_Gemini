import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import logging
import glob
import os
from .text_processor import DocumentProcessor
from .document_store import DocumentStore
from .ranking import DocumentRanker
from .question_classifier import QuestionClassifier
from .multi_query import MultiQueryRetriever
from .rag_fusion import RAGFusionRetriever

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
        self.multi_query = MultiQueryRetriever(self.api_key)
        self.rag_fusion = RAGFusionRetriever(self.api_key)

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
        template = """Sử dụng các đoạn ngữ cảnh sau để trả lời câu hỏi ở cuối. Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết, đừng cố tạo ra câu trả lời.

            Ngữ cảnh có thể bằng tiếng Anh hoặc tiếng Việt, nhưng bạn phải LUÔN trả lời bằng tiếng Việt và đảm bảo dịch chính xác các khái niệm khoa học.
            
            {context}
            Câu hỏi: {question}
            
            Role: Bạn là chemgenie chatbot AI, một chatbot hỗ trợ hóa học.
            Quy tắc định dạng:
            1. Đảm bảo tất cả công thức hóa học sử dụng ký hiệu chỉ số dưới đúng (ví dụ: viết CH₄ thay vì CH4)
            2. Sử dụng → cho mũi tên phản ứng (thay vì ->)
            3. Sử dụng ⇌ cho phản ứng thuận nghịch (thay vì <->)
            4. Định dạng câu trả lời với khoảng cách và ngắt dòng phù hợp:
               - Sử dụng ngắt dòng kép giữa các đoạn văn
               - Sử dụng dấu chấm đầu dòng (•) cho danh sách
               - Sử dụng in đậm (**) cho các thuật ngữ hoặc khái niệm quan trọng
               - Sử dụng thụt lề phù hợp cho phương trình hóa học
            Quy tắc trả lời:
            1. Trả lời bằng tiếng Việt rõ ràng, dễ hiểu
            2. ĐẶC BIỆT QUAN TRỌNG: Giữ nguyên danh pháp hóa học giống trong ngữ cảnh ở cả câu hỏi và các đáp án (danh pháp hóa học tiếng anh, IUPAC)
            3. Nếu ngữ cảnh bằng tiếng Anh, hãy dịch sang tiếng Việt nhưng giữ nguyên các thuật ngữ khoa học và công thức hóa học
            Câu trả lời hữu ích:"""
        
        prompt = PromptTemplate.from_template(template)
        
        self.qa_chain = (
            {"context": self.vector_index, "question": RunnablePassthrough()} 
            | prompt 
            | self.model 
            | StrOutputParser()
        )

    def clean_context(self, text):
        """Làm sạch và format lại ngữ cảnh"""
        # Loại bỏ khoảng trắng thừa
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Loại bỏ khoảng trắng thừa và các ký tự đặc biệt
            line = ' '.join(line.split())
            if line and not line.isspace():
                cleaned_lines.append(line)
        
        # Gộp các dòng thành đoạn văn
        paragraphs = []
        current_paragraph = []
        
        for line in cleaned_lines:
            if line.strip() in ['○', '•']:  # Markers for new paragraph
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                continue
            current_paragraph.append(line)
            
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            
        return '\n\n'.join(paragraphs)

    def ask_question(self, question):
        logging.info(f"Nhận được câu hỏi: {question}")
        try:
            # Phân loại câu hỏi
            if self.question_classifier.is_conversational(question):
                logging.info("Phát hiện câu hỏi giao tiếp, sử dụng LLM trực tiếp")
                return self.question_classifier.get_conversation_response(question)
            
            # Xử lý câu hỏi chuyên môn bằng RAG
            logging.info("Đang tìm kiếm tài liệu liên quan...")
            # retrieved_docs = self.multi_query.retrieve(question, self.vector_index)
            # retrieved_docs = self.vector_index.invoke(question)  
            doc_scores = self.rag_fusion.retrieve(question, self.vector_index)
    # Chỉ lấy documents (bỏ scores)
            retrieved_docs = [doc for doc, score in doc_scores]   
            logging.info(f"Tìm thấy {len(retrieved_docs)} tài liệu liên quan")
            
            logging.info("Bắt đầu rerank tài liệu...")
            reranked_docs = self.ranker.rerank_documents(question, retrieved_docs)
            logging.info("Hoàn thành rerank tài liệu")
            
            # Xử lý và làm sạch ngữ cảnh
            contexts = [self.clean_context(doc.page_content) for doc in reranked_docs]
            context = "\n\n".join(contexts)
            print(context)
            logging.info("Đang tạo câu trả lời...")
            answer = self.qa_chain.invoke(question)
            
            if not answer or answer.strip() == "":
                logging.warning("Không thể tạo câu trả lời")
                return "Xin lỗi, tôi không thể tạo câu trả lời. Vui lòng thử lại."
            
            logging.info("Đã tạo xong câu trả lời")
            return answer
            
        except Exception as e:
            logging.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")
            return "Đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại."