from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any
import logging

class GradeHallucinations(BaseModel):
    """Đánh giá hallucination trong câu trả lời"""
    binary_score: str = Field(description="Câu trả lời có dựa trên facts không, 'yes' hoặc 'no'")

class GradeAnswer(BaseModel):
    """Đánh giá câu trả lời có giải quyết câu hỏi không"""
    binary_score: str = Field(description="Câu trả lời có giải quyết câu hỏi không, 'yes' hoặc 'no'")

class QualityChecker:
    def __init__(self, llm, key_manager):
        self.llm = llm
        self.key_manager = key_manager
        self.setup_graders()
        
    def setup_graders(self):
        # Hallucination Grader
        system_hallucination = """Bạn là grader đánh giá xem câu trả lời của LLM có dựa trên facts được cung cấp hay không.
        Cho điểm binary 'yes' hoặc 'no'. 'Yes' nghĩa là câu trả lời được hỗ trợ bởi facts."""
        
        self.hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system_hallucination),
            ("human", "Facts: \n\n {documents} \n\n Câu trả lời LLM: {generation}")
        ])
        
        # Answer Grader
        system_answer = """Bạn là grader đánh giá xem câu trả lời có giải quyết câu hỏi hay không.
        Cho điểm binary 'yes' hoặc 'no'. 'Yes' nghĩa là câu trả lời giải quyết được câu hỏi."""
        
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system_answer),
            ("human", "Câu hỏi: \n\n {question} \n\n Câu trả lời LLM: {generation}")
        ])
        
        # Question Rewriter
        system_rewrite = """Bạn là question rewriter chuyển đổi câu hỏi thành phiên bản tốt hơn để tối ưu cho vectorstore retrieval.
        Hãy xem xét ý định ngữ nghĩa cơ bản. Chỉ cần trả về câu hỏi, không cần giải thích."""
        
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", system_rewrite),
            ("human", "Câu hỏi ban đầu: \n\n {question} \n Hãy tạo câu hỏi cải tiến.")
        ])

    def check_hallucination(self, documents: str, generation: str) -> bool:
        """Kiểm tra hallucination trong câu trả lời"""
        try:
            self.llm.google_api_key = self.key_manager.get_api_key()
            structured_llm = self.llm.with_structured_output(GradeHallucinations)
            hallucination_grader = self.hallucination_prompt | structured_llm
            result = hallucination_grader.invoke({
                "documents": documents,
                "generation": generation
            })
            return result.binary_score == "yes"
        except Exception as e:
            logging.error(f"Lỗi khi kiểm tra hallucination: {str(e)}")
            return False

    def check_answer_quality(self, question: str, generation: str) -> bool:
        """Kiểm tra chất lượng câu trả lời"""
        try:
            self.llm.google_api_key = self.key_manager.get_api_key()
            structured_llm = self.llm.with_structured_output(GradeAnswer)
            answer_grader = self.answer_prompt | structured_llm
            result = answer_grader.invoke({
                "question": question,
                "generation": generation
            })
            return result.binary_score == "yes"
        except Exception as e:
            logging.error(f"Lỗi khi kiểm tra chất lượng câu trả lời: {str(e)}")
            return False

    def rewrite_question(self, question: str) -> str:
        """Viết lại câu hỏi để tối ưu retrieval"""
        try:
            self.llm.google_api_key = self.key_manager.get_api_key()
            question_rewriter = self.rewrite_prompt | self.llm | StrOutputParser()
            return question_rewriter.invoke({"question": question})
        except Exception as e:
            logging.error(f"Lỗi khi viết lại câu hỏi: {str(e)}")
            return question 