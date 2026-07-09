import os
import sys
from dotenv import load_dotenv

# Ensure we're in the right directory and environment is loaded
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
load_dotenv()

from backend.llm.agent import DataWhispererAgent
from backend.services.file_service import FileService
from backend.services.chat_service import ChatService
from backend.llm.providers import create_default_provider
from backend.llm.chains.query_chain import QueryChain
from backend.llm.chains.output_parser import OutputParser
from backend.llm.prompts.registry import PromptRegistry
from backend.sandbox.executor import SandboxExecutor

def run_test():
    print("Initializing test...")
    
    # Initialize components
    provider = create_default_provider()
    sandbox = SandboxExecutor()
    query_chain = QueryChain(
        provider=provider,
        output_parser=OutputParser(),
        prompt_registry=PromptRegistry()
    )
    
    agent = DataWhispererAgent(
        provider=provider,
        sandbox=sandbox,
        query_chain=query_chain
    )
    
    file_service = FileService()
    
    # 1. Register the dataset
    csv_path = "sales_data.csv"
    print(f"Loading dataset: {csv_path}")
    
    with open(csv_path, "rb") as f:
        content = f.read()
    upload_resp = file_service.upload_file(filename="sales_data.csv", content=content)
    metadata = file_service.get_file_metadata(upload_resp.file_id)
    agent.register_dataset(metadata)
    
    print(f"Dataset registered. Columns: {metadata.column_names}")
    
    # 2. Ask a question
    session_id = "test-session-123"
    question = "What is the total sales by Region? Show me a bar chart."
    print(f"Asking question: {question}")
    
    response = agent.process_question(
        session_id=session_id,
        file_id=metadata.file_id,
        question=question,
        csv_path=metadata.stored_path,
        file_metadata=metadata
    )
    
    # 3. Print results
    print("\n" + "="*50)
    print(f"SUCCESS: {response.success}")
    print(f"RESULT TYPE: {response.result_type}")
    print(f"PROVIDER USED: {response.provider_used}")
    print(f"FALLBACK USED: {response.fallback_used}")
    if response.fallback_used:
        print(f"FALLBACK REASON: {response.fallback_reason}")
    print("="*50)
    print("GENERATED CODE:")
    print(response.code)
    print("="*50)
    print("EXPLANATION:")
    print(response.explanation)
    print("="*50)
    if response.chart_path:
        print(f"CHART PATH: {response.chart_path}")
        print(f"CHART REASONING: {response.chart_explanation}")
    else:
        print("RESULT DATA:")
        print(response.result_data)
        
if __name__ == "__main__":
    run_test()
