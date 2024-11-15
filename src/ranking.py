from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
from langchain_core.documents import Document
import logging
from rank_bm25 import BM25Okapi
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
# File này để rerank các tài liệu dựa trên độ liên quan với câu hỏi
class RatingScore(BaseModel):
    relevance_score: float = Field(..., description="Điểm đánh giá độ liên quan của tài liệu với câu hỏi.")

class DocumentRanker:
    def __init__(self, llm, max_workers=4, cache_size=100, pre_filter_k=10):
        self.llm = llm
        self.max_workers = max_workers
        self.pre_filter_k = pre_filter_k  # Số lượng document giữ lại sau BM25
        self.prompt_template = PromptTemplate(
            input_variables=["query", "doc"],
            template="""On a scale of 1-10, rate the relevance of the following document to the query. Consider the specific context and intent of the query, not just keyword matches.
            Query: {query}
            Document: {doc}
            Relevance Score:"""
        )

    def _create_bm25(self, docs):
        # Tạo BM25 index
        tokenized_docs = [doc.page_content.lower().split() for doc in docs]
        return BM25Okapi(tokenized_docs)

    @lru_cache(maxsize=100)
    def _get_cached_score(self, query: str, doc_content: str) -> float:
        llm_chain = self.prompt_template | self.llm.with_structured_output(RatingScore)
        return float(llm_chain.invoke({"query": query, "doc": doc_content}).relevance_score)

    def rerank_documents(self, query: str, docs: List[Document], top_n: int = 3) -> List[Document]:
        if len(docs) == 0:
            return []

        logging.info(f"Bắt đầu rerank {len(docs)} tài liệu...")
        
        # Bước 1: Sử dụng BM25 để lọc sơ bộ
        bm25 = self._create_bm25(docs)
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Lấy top-k documents từ BM25
        pre_filtered_indices = sorted(range(len(docs)), 
                                   key=lambda i: bm25_scores[i], 
                                   reverse=True)[:self.pre_filter_k]
        pre_filtered_docs = [docs[i] for i in pre_filtered_indices]
        
        logging.info(f"Đã lọc còn {len(pre_filtered_docs)} tài liệu qua BM25")

        # Bước 2: Sử dụng LLM để rerank các document đã lọc
        scored_docs = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {
                executor.submit(self._get_cached_score, 
                              query, 
                              doc.page_content): (doc, idx) 
                for idx, doc in enumerate(pre_filtered_docs)
            }
            
            for future in as_completed(future_to_doc):
                doc, idx = future_to_doc[future]
                try:
                    score = future.result()
                    scored_docs.append((doc, score))
                    logging.info(f"\nDocument {idx + 1}:")
                    logging.info(f"Score: {score}")
                    logging.info(f"Preview: {doc.page_content[:200]}...")
                except Exception as e:
                    logging.error(f"Lỗi khi đánh giá document {idx + 1}: {str(e)}")
                    scored_docs.append((doc, 0))

        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        # Log kết quả cuối cùng
        logging.info("\nKết quả rerank cuối cùng:")
        for idx, (doc, score) in enumerate(reranked_docs[:top_n]):
            logging.info(f"\nTop {idx + 1}:")
            logging.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
            logging.info(f"Score: {score}")

        return [doc for doc, _ in reranked_docs[:top_n]]