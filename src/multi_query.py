from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.load import dumps, loads
from typing import List
from langchain_core.documents import Document

class MultiQueryRetriever:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0
        )
        
        self.query_prompt = ChatPromptTemplate.from_template(
            """Bạn là một AI assistant. Nhiệm vụ của bạn là tạo ra năm phiên bản khác nhau 
            của câu hỏi được đưa ra để tìm kiếm các tài liệu liên quan từ cơ sở dữ liệu vector. 
            Bằng cách tạo ra nhiều góc nhìn khác nhau về câu hỏi của người dùng, mục tiêu của bạn 
            là giúp vượt qua một số hạn chế của tìm kiếm dựa trên khoảng cách.
            Đưa ra các câu hỏi thay thế được phân tách bằng dòng mới.
            
            Câu hỏi gốc: {question}"""
        )

        # Tạo chain để generate queries
        self.generate_queries = (
            self.query_prompt 
            | self.llm 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

    def get_unique_docs(self, documents: List[List[Document]]) -> List[Document]:
        """Lấy union của các documents đã retrieve, loại bỏ trùng lặp"""
        # Flatten list và chuyển documents thành strings
        flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
        # Loại bỏ trùng lặp
        unique_docs = list(set(flattened_docs))
        # Chuyển lại thành Document objects
        return [loads(doc) for doc in unique_docs]

    def retrieve(self, question: str, retriever) -> List[Document]:
        """Thực hiện multi-query retrieval sử dụng chain"""
        # Tạo retrieval chain
        retrieval_chain = (
            self.generate_queries 
            | retriever.map()  # map() sẽ áp dụng retriever cho mỗi query
            | self.get_unique_docs
        )
        
        # Thực thi chain
        return retrieval_chain.invoke({"question": question})