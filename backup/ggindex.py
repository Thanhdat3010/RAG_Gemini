import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyD3SQ-8fHNjPnEGn4gegLk57JNNQO8U8lI"

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import TextSplitter
from langchain_community.vectorstores import FAISS

# Khởi tạo đối tượng LineTextSplitter
class LineTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split('\n')

# Định nghĩa class kế thừa TextSplitter để chia văn bản theo dòng
text_splitter = LineTextSplitter()

# Danh sách các tệp cần đọc
files = ['data\\about_us.txt', 'data\\bangtuanhoan_db.txt', 'data\\danhphaphoahoc.txt', 'data\\lythuyet_hoa12.txt']

# Khởi tạo một danh sách để lưu FAISS vectorstores cho mỗi tệp
faiss_dbs = {}

for file_path in files:
    # Đọc nội dung của tệp
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        # Chia văn bản thành các đoạn
        documents = text_splitter.split_text(text)
        
        # Tạo FAISS vectorstore cho tệp hiện tại
        faiss_db = FAISS.from_texts(documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
        
        # Lưu FAISS vectorstore vào thư mục googleai_index với tên tệp tương ứng
        base_filename = os.path.basename(file_path).replace('.txt', '')  # Lấy tên tệp mà không có đuôi mở rộng
        faiss_db.save_local(f"googleai_index\\{base_filename}_index")

print("All FAISS vectorstores have been saved.")
