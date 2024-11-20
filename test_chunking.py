import glob
import os
from src.text_processor import DocumentProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings

FOLDER_PATH = "data"

def test_document_chunking():
    """Hàm test chunking với các tham số khác nhau"""
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
        
        choice = input("\nChọn số thứ tự file muốn test chunking (0 để thoát): ")
        if choice == "0":
            break
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(all_files):
                file_path = all_files[idx]
                test_chunking_parameters(file_path)
            else:
                print("❌ Số thứ tự không hợp lệ!")
        except ValueError:
            print("❌ Vui lòng nhập số!")

def test_chunking_parameters(file_path):
    """Test chunking với các tham số khác nhau cho một file"""
    print(f"\n{'='*100}")
    print(f"📄 File: {os.path.basename(file_path)}")
    print(f"📍 Đường dẫn đầy đủ: {file_path}")
    print(f"📊 Kích thước: {os.path.getsize(file_path)/1024:.2f} KB")
    print(f"{'='*100}\n")

    try:
        # Đọc nội dung file
        loader = DocumentProcessor.load_document(file_path)
        pages = loader.load_and_split()
        content = "\n\n".join(str(p.page_content) for p in pages)
        
        # Thêm các tham số được đề xuất
        suggested_params = [
            {"chunk_size": 500, "chunk_overlap": 50},
            {"chunk_size": 1000, "chunk_overlap": 100},
            {"chunk_size": 2000, "chunk_overlap": 200},
        ]
        
        print("\n📊 THAM SỐ ĐỀ XUẤT:")
        for i, params in enumerate(suggested_params, 1):
            print(f"{i}. chunk_size: {params['chunk_size']}, chunk_overlap: {params['chunk_overlap']}")
        
        while True:
            print("\n🔧 THIẾT LẬP THAM SỐ CHUNKING:")
            try:
                chunk_size = int(input("Nhập chunk_size (VD: 1000): "))
                chunk_overlap = int(input("Nhập chunk_overlap (VD: 50): "))
                
                if chunk_overlap >= chunk_size:
                    print("❌ chunk_overlap phải nhỏ hơn chunk_size!")
                    continue
                
                # Thực hiện chunking với tham số mới
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                chunks = text_splitter.split_text(content)
                
                print(f"\n📊 KẾT QUẢ CHUNKING:")
                print(f"Tổng số chunk: {len(chunks)}")
                print(f"{'Index':<8} {'Độ dài':<10} {'Overlap':<10} {'Preview':<50}")
                print("-" * 78)
                
                for idx, chunk in enumerate(chunks):
                    # Tính overlap với chunk tiếp theo
                    overlap_text = ""
                    if idx < len(chunks) - 1:
                        next_chunk = chunks[idx + 1]
                        overlap_text = find_overlap(chunk, next_chunk)
                    
                    # Hiển thị preview của chunk (50 ký tự đầu)
                    preview = chunk[:50].replace('\n', ' ').strip() + "..."
                    print(f"{idx:<8} {len(chunk):<10} {len(overlap_text) if overlap_text else 0:<10} {preview:<50}")
                
                # Đánh giá chất lượng chunking
                avg_chunk_size = sum(len(chunk) for chunk in chunks) / len(chunks)
                avg_overlap = sum(len(find_overlap(chunks[i], chunks[i+1])) 
                                for i in range(len(chunks)-1)) / (len(chunks)-1) if len(chunks) > 1 else 0
                
                print(f"\n📈 ĐÁNH GIÁ CHẤT LƯỢNG CHUNKING:")
                print(f"- Số lượng chunk: {len(chunks)}")
                print(f"- Độ dài chunk trung bình: {avg_chunk_size:.1f} ký tự")
                print(f"- Overlap trung bình: {avg_overlap:.1f} ký tự")
                print(f"- Đánh giá: ", end="")
                
                if avg_chunk_size < 200:
                    print("❌ Chunk quá ngắn, nên tăng chunk_size")
                elif avg_chunk_size > 2000:
                    print("❌ Chunk quá dài, nên giảm chunk_size")
                elif avg_overlap < chunk_overlap * 0.5:
                    print("⚠️ Overlap thực tế thấp hơn mong đợi")
                else:
                    print("✅ Tham số phù hợp")

                while True:
                    chunk_idx = input("\nNhập số thứ tự chunk muốn xem (-1 để thử tham số khác): ")
                    if chunk_idx == "-1":
                        break
                    
                    try:
                        chunk_idx = int(chunk_idx)
                        if 0 <= chunk_idx < len(chunks):
                            print(f"\n📝 NỘI DUNG CHUNK {chunk_idx}:")
                            print("-" * 100)
                            print(chunks[chunk_idx])
                            print("-" * 100)
                        else:
                            print("❌ Số thứ tự chunk không hợp lệ!")
                    except ValueError:
                        print("❌ Vui lòng nhập số!")
                
            except ValueError:
                print("❌ Vui lòng nhập số cho các tham số!")
                
    except Exception as e:
        print(f"❌ Lỗi khi xử lý file: {str(e)}")

def find_overlap(text1, text2):
    """Tìm phần overlap giữa hai đoạn text"""
    # Tìm đoạn text chung dài nhất ở cuối text1 và đầu text2
    min_len = min(len(text1), len(text2))
    for i in range(min_len, 0, -1):
        if text1[-i:] == text2[:i]:
            return text1[-i:]
    return ""

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    test_document_chunking() 