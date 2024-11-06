from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
import json
import cv2
import numpy as np
from fer import FER
from deepface import DeepFace
import logging
import base64

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

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

def lythuyet_hoa12(info: str) -> str:
    db = FAISS.load_local("googleai_index/lythuyet_hoa12_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    results = db.similarity_search(info, k=1)
    docs = [{"content": doc.page_content} for doc in results]
    return json.dumps(docs, ensure_ascii=False)

def ester(info: str) -> str:
    db = FAISS.load_local("googleai_index/ester_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    results = db.similarity_search(info, k=1)
    docs = [{"content": doc.page_content} for doc in results]
    return json.dumps(docs, ensure_ascii=False)

def detect_emotion(image_data):
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    fer_emotion = "neutral"
    deepface_emotion = "neutral"
    
    try:
        # FER detection
        detector = FER()
        fer_emotion, fer_score = detector.top_emotion(img)
        if not fer_emotion:
            fer_emotion = "neutral"
        
        # DeepFace detection
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        deepface_emotion = result[0]['dominant_emotion']
        
        # Combine results
        if fer_emotion == deepface_emotion:
            final_emotion = fer_emotion
        else:
            # If different, prefer DeepFace but with a higher threshold
            deepface_score = result[0]['emotion'][deepface_emotion]
            if deepface_score > 0.6:  # Adjust this threshold as needed
                final_emotion = deepface_emotion
            else:
                final_emotion = fer_emotion
        
        logging.debug(f"FER emotion: {fer_emotion}, DeepFace emotion: {deepface_emotion}, Final: {final_emotion}")
        return final_emotion
    except Exception as e:
        logging.error(f"Error detecting emotion: {str(e)}")
        return "neutral"

tools = [about_us, about_periodic, research_iupac, lythuyet_hoa12, ester]

available_tools = {
    "about_us": about_us,
    "about_periodic": about_periodic,
    "research_iupac": research_iupac,
    "lythuyet_hoa12": lythuyet_hoa12,
    "ester": ester
}

model = genai.GenerativeModel(model_name="gemini-1.5-flash", tools=tools)

history = [
    {
        "role": "user",
        "parts": [
            """Bạn là trợ lý ảo thông minh hỗ trợ người dùng trang web Chemgenie. Nhiệm vụ chính của bạn là:
            1. Nếu câu hỏi liên quan đến thông tin về đội ngũ hoặc website, sử dụng công cụ `about_us`.
            2. Nếu câu hỏi liên quan đến nguyên tố trong bảng tuần hoàn, sử dụng công cụ `about_periodic`.
            3. Nếu câu hỏi liên quan đến danh pháp IUPAC, sử dụng công cụ `research_iupac`.
            4. Nếu câu hỏi liên quan đến lý thuyết hóa học lớp 12, sử dụng công cụ `lythuyet_hoa12`.
             5. Nếu câu hỏi liên quan đến ester, sử dụng công cụ `ester`.
            Lưu ý:
            - Nếu không phải hóa thì cứ trả lời tôi không biết theo ý bạn.
            - Nếu không thể tìm thấy thông tin, hãy trả lời lịch sự rằng không có thông tin khả dụng.
            - Đừng đưa ra thông tin không chính xác hoặc ngoài phạm vi dữ liệu.
            - Ưu tiên trả lời ngắn gọn, dễ hiểu và chính xác.
            - QUAN TRỌNG: Các chất hóa học thì phải sử dụng theo danh pháp hóa học IUPAC (danh pháp hóa học tiếng anh)
            - Phải sử dụng tiếng việt để trả lời người dùng
            """,
        ],
    },
]

chat = model.start_chat(history=history)

@app.route('/chat', methods=['POST'])
def chat_with_bot():
    user_input = request.form.get('message')
    image = request.files.get('image')
    
    logging.debug(f"Received message: {user_input}")
    logging.debug(f"Image received: {image is not None}")
    
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    
    emotion = "neutral"
    if image:
        image_data = image.read()
        emotion = detect_emotion(image_data)
        logging.debug(f"Detected emotion: {emotion}")
    
    emotion_prompt = get_emotion_prompt(emotion)
    user_message = f"{emotion_prompt} User's message: {user_input}"
    
    response = chat.send_message(user_message)

    responses = {}

    for part in response.parts:
        if fn := part.function_call:
            function_name = fn.name
            function_args = fn.args.get('info', '')
            function_to_call = available_tools.get(function_name)
            
            if function_to_call:
                function_response = function_to_call(function_args)
                responses[function_name] = function_response

    if responses:
        response_parts = [
            genai.protos.Part(function_response=genai.protos.FunctionResponse(name=fn, response={"result": val}))
            for fn, val in responses.items()
        ]
        response = chat.send_message(response_parts)

    final_response = response.text

    response_data = {"response": final_response, "detected_emotion": emotion}
    logging.debug(f"Sending response: {response_data}")
    return jsonify(response_data)

def get_emotion_prompt(emotion):
    emotion_prompts = {
        "happy": "The user seems happy. Respond in a cheerful and positive manner. ",
        "sad": "The user appears sad. Show empathy and offer encouragement in your response. ",
        "angry": "The user seems angry. Respond calmly and try to address their concerns. ",
        "surprised": "The user looks surprised. Acknowledge their reaction and provide clear information. ",
        "neutral": "The user's emotion is neutral. Respond in a balanced and informative way. "
    }
    return emotion_prompts.get(emotion, "Respond appropriately based on the user's message. ")

@app.route('/test_emotion', methods=['POST'])
def test_emotion():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "Image is required"}), 400
    
    image_data = image.read()
    emotion = detect_emotion(image_data)
    return jsonify({"detected_emotion": emotion})

if __name__ == '__main__':
    app.run(debug=True)