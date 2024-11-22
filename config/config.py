import os
import random
from typing import List
import logging

class APIKeyManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.call_count = 0
    
    def get_api_key(self) -> str:
        self.call_count += 1
        key = random.choice(self.api_keys)
        # Chỉ hiện 8 ký tự đầu của key để bảo mật
        masked_key = key[:8] + "..." 
        logging.info(f"API Call #{self.call_count} - Using key: {masked_key}")
        return key

# List các API keys
GOOGLE_API_KEYS = [
    "AIzaSyCdm8-0p8SQREhPDicyywkDeDbMVtMmhNo",
    "AIzaSyBc1fHj2tGSwmVraM39ZXzFjvy_qubMct8",
]

# Tạo instance của API Key Manager
key_manager = APIKeyManager(GOOGLE_API_KEYS)

FOLDER_PATH = "data"
VECTOR_STORE_PATH = "vector_store"
FILE_HASH_PATH = "file_hashes/file_hashes.pkl"