import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyD3SQ-8fHNjPnEGn4gegLk57JNNQO8U8lI"
import PyPDF2
from docx import Document
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
files = [
    'data\\about_us.txt',
    'data\\bangtuanhoan_db.txt',
    'data\\danhphaphoahoc.txt',
    'data\\lythuyet_hoa12.txt',
    'data\\ester.docx',  # Tệp Word
]

# Khởi tạo một danh sách để lưu FAISS vectorstores cho mỗi tệp
faiss_dbs = {}

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_docx(file_path):
    doc = Document(file_path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return '\n'.join(text)

def read_pdf(file_path):
    text = []
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text())
    return '\n'.join(text)

for file_path in files:
    if file_path.endswith('.txt'):
        text = read_txt(file_path)
    elif file_path.endswith('.docx'):
        text = read_docx(file_path)
    elif file_path.endswith('.pdf'):
        text = read_pdf(file_path)
    else:
        continue  # Bỏ qua nếu không phải định dạng được hỗ trợ

    # Chia văn bản thành các đoạn
    documents = text_splitter.split_text(text)

    # Tạo FAISS vectorstore cho tệp hiện tại
    faiss_db = FAISS.from_texts(documents, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

    # Lưu FAISS vectorstore vào thư mục googleai_index với tên tệp tương ứng
    base_filename = os.path.basename(file_path).replace('.txt', '').replace('.docx', '').replace('.pdf', '')  # Lấy tên tệp mà không có đuôi mở rộng
    faiss_db.save_local(f"googleai_index\\{base_filename}_index")

print("All FAISS vectorstores have been saved.")
