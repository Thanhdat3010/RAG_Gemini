from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.load import dumps, loads
from typing import List
from langchain_core.documents import Document

class RAGFusionRetriever:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0
        )
        
        # RAG-Fusion: Related
        self.query_prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant that generates multiple search queries based on a single input query.

                Generate multiple search queries related to: {question}

                Output (4 queries):"""
        )

        # Tạo chain để generate queries
        self.generate_queries = (
            self.query_prompt 
            | self.llm 
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )

    def translate_query(self, question: str) -> str:
        """Dịch câu hỏi sang tiếng Anh, giữ nguyên các thuật ngữ hóa học"""
        translate_prompt = """Translate this chemistry question to English. 
        Keep all chemical formulas, IUPAC names, and technical terms unchanged.
        
        Question: {question}
        
        English translation:"""
        
        english_query = self.llm.invoke(translate_prompt.format(question=question))
        return english_query

    def reciprocal_rank_fusion(self, results: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
            and an optional parameter k used in the RRF formula """
        
        fused_scores = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        reranked_results = [
            (loads(doc), score)  # Trả về tuple (document, score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        # Return the reranked results as a list of tuples
        return reranked_results

    def retrieve(self, question: str, retriever) -> List[tuple]:
        """Thực hiện RAG-Fusion retrieval"""
        # Translate query first
        english_question = self.translate_query(question)
        
        # Use translated query for retrieval
        retrieval_chain = (
            self.generate_queries  # This will now work with English query
            | retriever.map()
            | self.reciprocal_rank_fusion
        )
        return retrieval_chain.invoke({"question": english_question}) 