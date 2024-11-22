from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
# File này để phân loại câu hỏi có phải là giao tiếp thông thường hay câu hỏi chuyên môn cần RAG
class QuestionClassifier:
    def __init__(self, key_manager):
        self.key_manager = key_manager
        self.classifier = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.key_manager.get_api_key(),
            temperature=0
        )
        self.load_prompts()
    
    def load_prompts(self):
        """Load prompts from config file"""
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'prompts.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
                self.classification_prompt = prompts.get('classification_prompt', self.default_classification_prompt())
                self.conversation_prompt = prompts.get('conversation_prompt', self.default_conversation_prompt())
        except FileNotFoundError:
            self.classification_prompt = self.default_classification_prompt()
            self.conversation_prompt = self.default_conversation_prompt()
    
    def default_classification_prompt(self):
        return """Hãy phân loại câu hỏi sau là câu hỏi giao tiếp thông thường hay câu hỏi chuyên môn cần tra cứu tài liệu.
        Trả lời "True" nếu là câu hỏi giao tiếp thông thường (ví dụ: chào hỏi, hỏi thăm, giới thiệu, cảm xúc...).
        Trả lời "False" nếu là câu hỏi cần tra cứu thông tin hoặc kiến thức (ví dụ: hỏi về địa điểm, món ăn, lịch sử, văn hóa...).
        Lưu ý: Nếu câu hỏi liên quan đến thông tin thực tế, ngay cả khi được hỏi một cách thân mật, vẫn phải trả lời "False".
        Chỉ trả lời "True" hoặc "False", không giải thích thêm.
        
        Câu hỏi: {question}"""
    
    def default_conversation_prompt(self):
        return """Bạn là một trợ lý hóa học Chemgenie AI thân thiện. Hãy trả lời câu hỏi sau một cách ngắn gọn và thân thiện:

        {question}"""
    
    def is_conversational(self, question):
        """Determine if a question is conversational"""
        self.classifier.google_api_key = self.key_manager.get_api_key()
        response = self.classifier.invoke(
            self.classification_prompt.format(question=question)
        )
        return response.content.strip().lower() == "true"
    
    def get_conversation_response(self, question):
        """Get response for conversational questions"""
        self.classifier.google_api_key = self.key_manager.get_api_key()
        return self.classifier.invoke(
            self.conversation_prompt.format(question=question)
        ).content 