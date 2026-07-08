import sys
import os
import io
import time
from pathlib import Path
from dotenv import load_dotenv

# Load env to ensure Grok API key is read
load_dotenv()

# Force logs so we can see what's happening
os.environ["LOG_LEVEL"] = "INFO"

from backend.llm.factory import create_agent
from backend.services.file_service import FileService

def test_agent():
    print("Initializing FileService...")
    file_service = FileService()
    
    csv_path = Path("test_products.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return
        
    content = csv_path.read_bytes()
    
    print(f"Uploading {csv_path.name}...")
    upload_response = file_service.upload_file(
        filename=csv_path.name,
        content=content,
    )
    
    metadata = file_service.get_file_metadata(upload_response.file_id)
    print(f"File ID: {metadata.file_id}")
    
    print("Initializing DataWhispererAgent...")
    agent = create_agent()
    
    session_id = "test-session-123"
    question = "What is the most expensive product?"
    
    print(f"Processing question: '{question}'...")
    
    result = agent.process_question(
        session_id=session_id,
        file_id=metadata.file_id,
        question=question,
        csv_path=metadata.stored_path,
        file_metadata=metadata,
    )
    
    print("\n" + "="*50)
    print(f"Success: {result.success}")
    print(f"Provider Used: {result.provider_used}")
    print(f"Model Used: {result.model_used}")
    print(f"Fallback Used: {result.fallback_used}")
    if result.fallback_used:
        print(f"Fallback Reason: {result.fallback_reason}")
    print("="*50)
    print("Generated Code:")
    print(result.code)
    print("="*50)
    print("Output Data:")
    print(result.result_data)
    print("="*50)

if __name__ == "__main__":
    test_agent()
