import glob
import os
import time
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from docling.document_converter import DocumentConverter
import warnings
from difflib import SequenceMatcher

FOLDER_PATH = "data"  # thư mục chứa dữ liệu của bạn

def calculate_similarity(text1, text2):
    """Tính độ tương đồng giữa hai văn bản"""
    return SequenceMatcher(None, text1, text2).ratio()

def test_document_loaders():
    """Hàm test so sánh docling và loader truyền thống cho tất cả file trong thư mục data"""
    # Lấy danh sách tất cả các file
    pdf_files = glob.glob(os.path.join(FOLDER_PATH, "*.pdf"))
    word_files = glob.glob(os.path.join(FOLDER_PATH, "*.docx*"))
    txt_files = glob.glob(os.path.join(FOLDER_PATH, "*.txt"))
    all_files = pdf_files + word_files + txt_files

    if not all_files:
        print("❌ Không tìm thấy file nào trong thư mục data")
        return

    while True:
        print(f"\n🗂️ Danh sách file tìm thấy ({len(all_files)} file):")
        for idx, file in enumerate(all_files, 1):
            print(f"{idx}. {os.path.basename(file)}")
        
        choice = input("\nChọn số thứ tự file muốn test (0 để thoát): ")
        if choice == "0":
            break
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(all_files):
                file_path = all_files[idx]
                test_single_file(file_path)
            else:
                print("❌ Số thứ tự không hợp lệ!")
        except ValueError:
            print("❌ Vui lòng nhập số!")

def test_single_file(file_path):
    """Test một file cụ thể"""
    print(f"\n{'='*100}")
    print(f"📄 File: {os.path.basename(file_path)}")
    print(f"📍 Đường dẫn đầy đủ: {file_path}")
    print(f"📊 Kích thước: {os.path.getsize(file_path)/1024:.2f} KB")
    print(f"{'='*100}\n")
    
    docling_text = ""
    traditional_text = ""
    docling_time = 0
    traditional_time = 0
    
    # Test docling
    print("🔍 THỬ NGHIỆM VỚI DOCLING:")
    start_time = time.time()
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        docling_text = result.document.export_to_markdown()
        docling_time = time.time() - start_time
        
        print(f"📝 Độ dài văn bản: {len(docling_text)} ký tự")
        print(f"⏱️ Thời gian xử lý: {docling_time:.2f} giây")
    except Exception as e:
        print(f"❌ Lỗi khi dùng Docling: {str(e)}")
    
    # Test loader truyền thống
    print("\n🔍 THỬ NGHIỆM VỚI LOADER TRUYỀN THỐNG:")
    start_time = time.time()
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension in ['.doc', '.docx']:
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            print(f"❌ Không hỗ trợ định dạng file: {file_extension}")
            return
            
        pages = loader.load_and_split()
        traditional_text = "\n\n".join(str(p.page_content) for p in pages)
        traditional_time = time.time() - start_time
        
        print(f"📝 Độ dài văn bản: {len(traditional_text)} ký tự")
        print(f"📄 Số trang/đoạn: {len(pages)}")
        print(f"⏱️ Thời gian xử lý: {traditional_time:.2f} giây")
    except Exception as e:
        print(f"❌ Lỗi khi dùng loader truyền thống: {str(e)}")
    
    # So sánh kết quả
    print("\n📊 KẾT QUẢ SO SÁNH:")
    print("-" * 50)
    if docling_text and traditional_text:
        similarity = calculate_similarity(docling_text, traditional_text)
        print(f"🔄 Độ tương đồng nội dung: {similarity:.2%}")
        print(f"⚡ Chênh lệch độ dài: {abs(len(docling_text) - len(traditional_text))} ký tự")
        print(f"⏱️ Chênh lệch thời gian: {abs(docling_time - traditional_time):.2f} giây")
        
        if similarity < 0.5:
            print("\n⚠️ Cảnh báo: Độ tương đồng thấp, có thể có sự khác biệt lớn trong kết quả")
            
        print("\n💡 Đánh giá:")
        if docling_time < traditional_time:
            print("- Docling xử lý nhanh hơn")
        else:
            print("- Loader truyền thống xử lý nhanh hơn")
            
        if len(docling_text) > len(traditional_text):
            print("- Docling trích xuất được nhiều nội dung hơn")
        else:
            print("- Loader truyền thống trích xuất được nhiều nội dung hơn")
    else:
        print("❌ Không thể so sánh do một trong hai phương pháp bị lỗi")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    test_document_loaders()