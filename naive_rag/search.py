from typing import List
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")
client = OpenAI(api_key=OPENAI_API_KEY)

def search_documents(vectorstore, query: str, k: int = 5) -> List[str]:
    """
    Perform similarity search on the vector store using the provided query.
    """
    try:
        results = vectorstore.similarity_search(
            query,
            k=k
        )
        return [doc.page_content for doc in results]
    except Exception as e:
        print(f"Error performing similarity search: {str(e)}")
        return []

def get_answer(query: str, context: List[str]) -> str:
    """
    Generate answer using OpenAI API based on the query and context.
    """
    try:
        system_prompt = f'''You are an intelligent bot that answers questions based on the provided context.
        Context: {context}
        
        Please provide a clear and concise answer based on the context above.
        If the context doesn't contain enough information to answer the question, please say so.
        '''
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return "Sorry, I encountered an error while generating the answer." 