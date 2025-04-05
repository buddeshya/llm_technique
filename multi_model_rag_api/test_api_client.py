import requests
import json

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"Health check response: {response.json()}")
    
    # Test queries
    test_queries = [
        "What is multi-head attention?",
        "Explain the transformer architecture.",
        "What are the key components of the transformer model?",
        "How does the transformer handle sequence transduction?",
        "What are the advantages of the transformer over RNNs?"
    ]
    
    print("\nTesting queries...")
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        # Send query to API
        response = requests.post(
            f"{base_url}/query",
            json={"question": query}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result['answer']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
        
        print("-" * 80)

if __name__ == "__main__":
    test_api() 