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

class ChemGenieBot:
    def __init__(self, api_key, folder_path):
        self.api_key = api_key
        self.folder_path = folder_path
        self.setup_model()
        self.load_and_process_documents()
        self.setup_qa_chain()
        self.conversation_history = []

    def setup_model(self):
        genai.configure(api_key=self.api_key)
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.2,
            convert_system_message_to_human=True
        )

    def load_and_process_documents(self):
        texts = []
        
        # Tìm tất cả các file PDF, Word và TXT trong thư mục
        pdf_files = glob.glob(os.path.join(self.folder_path, "*.pdf"))
        word_files = glob.glob(os.path.join(self.folder_path, "*.docx*"))
        txt_files = glob.glob(os.path.join(self.folder_path, "*.txt"))
        all_files = pdf_files + word_files + txt_files
        
        if not all_files:
            raise ValueError("No supported documents found in the specified folder.")
        
        for file_path in all_files:
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
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )
        
        self.vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 5})

    def setup_qa_chain(self):
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

    def ask_question(self, question):
        self.conversation_history.append(f"User: {question}")
        
        context_with_history = "\n".join(self.conversation_history)
        result = self.qa_chain({"query": context_with_history})
        
        answer = result["result"]
        self.conversation_history.append(f"Bot: {answer}")
        
        return answer

def main():
    # Configure your API key and PDF path
    GOOGLE_API_KEY = "AIzaSyD3SQ-8fHNjPnEGn4gegLk57JNNQO8U8lI"
    FOLDER_PATH = "data"
    
    # Initialize the chatbot
    bot = ChemGenieBot(GOOGLE_API_KEY, FOLDER_PATH)
    
    # Example usage
    while True:
        question = input("Hỏi câu hỏi của bạn (hoặc 'quit' để thoát): ")
        if question.lower() == 'quit':
            break
        
        answer = bot.ask_question(question)
        print("\nTrả lời:", answer)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main() 