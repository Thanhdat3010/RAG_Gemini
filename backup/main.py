from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS để cho phép kết nối từ frontend
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import json


# file này là file gốc


app = Flask(__name__)
CORS(app)  # Cấu hình CORS để cho phép các request từ frontend React

# Cấu hình API key cho Google Generative AI
genai.configure(api_key="AIzaSyD3SQ-8fHNjPnEGn4gegLk57JNNQO8U8lI")
os.environ["GOOGLE_API_KEY"] = "AIzaSyD3SQ-8fHNjPnEGn4gegLk57JNNQO8U8lI"

# Định nghĩa các công cụ
def about_us(info: str) -> str:
    db = FAISS.load_local("googleai_index/about_us_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    results = db.similarity_search(info, k=1)
    docs = [{"content": doc.page_content} for doc in results]
    return json.dumps(docs, ensure_ascii=False)

def about_periodic(info: str) -> str:
    db = FAISS.load_local("googleai_index/bangtuanhoan_db_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    results = db.similarity_search(info, k=3)
    docs = [{"content": doc.page_content} for doc in results]
    return json.dumps(docs, ensure_ascii=False)

def research_iupac(info: str) -> str:
    db = FAISS.load_local("googleai_index/danhphaphoahoc_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    results = db.similarity_search(info, k=1)
    docs = [{"content": doc.page_content} for doc in results]
    return json.dumps(docs, ensure_ascii=False)

# Danh sách các hàm (công cụ) có sẵn cho chatbot
tools = [about_us, about_periodic, research_iupac]

# Tạo dictionary ánh xạ tên hàm với hàm tương ứng
available_tools = {
    "about_us": about_us,
    "about_periodic": about_periodic,
    "research_iupac": research_iupac
}

# Khởi tạo model Google Generative AI với tên model và danh sách công cụ
model = genai.GenerativeModel(model_name="gemini-1.5-flash", tools=tools)

# Tạo chatbot với system message để cấu hình chatbot
history = [
    {
        "role": "user",
        "parts": [
            """Bạn là trợ lý ảo thông minh hỗ trợ người dùng trang web Chemgenie. Nhiệm vụ chính của bạn là:
            1. Nếu câu hỏi liên quan đến thông tin về đội ngũ hoặc website, sử dụng công cụ `about_us`.
            2. Nếu câu hỏi liên quan đến nguyên tố trong bảng tuần hoàn, sử dụng công cụ `about_periodic`.
            3. Nếu câu hỏi liên quan đến danh pháp IUPAC, sử dụng công cụ `research_iupac`.
            
            Lưu ý:
            - Nếu không thể tìm thấy thông tin, hãy trả lời lịch sự rằng không có thông tin khả dụng.
            - Đừng đưa ra thông tin không chính xác hoặc ngoài phạm vi dữ liệu.
            - Ưu tiên trả lời ngắn gọn, dễ hiểu và chính xác.
            """,
        ],
    },
]

chat = model.start_chat(history=history)

# API để nhận tin nhắn từ frontend và trả phản hồi từ chatbot
@app.route('/chat', methods=['POST'])
def chat_with_bot():
    user_input = request.json.get('message')
    
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    
    # Gửi tin nhắn của người dùng cho chatbot và nhận phản hồi
    response = chat.send_message(user_input)

    # Tạo dictionary để lưu trữ kết quả từ các hàm
    responses = {}

    # Xử lý từng phần của phản hồi từ chatbot
    for part in response.parts:
        if fn := part.function_call:
            function_name = fn.name
            function_args = fn.args.get('info', '')
            function_to_call = available_tools.get(function_name)
            
            if function_to_call:
                function_response = function_to_call(function_args)
                responses[function_name] = function_response

    # Nếu có kết quả từ các hàm
    if responses:
        response_parts = [
            genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn, response={"result": val}))
            for fn, val in responses.items()
        ]
        response = chat.send_message(response_parts)

    # Trả phản hồi cuối cùng cho client (frontend ReactJS)
    return jsonify({"response": response.text})

if __name__ == '__main__':
    app.run(debug=True)