import os
from dotenv import load_dotenv
load_dotenv()

from agent import build_agent_graph
from langchain_core.messages import HumanMessage

def test_pipeline():
    print("🚀 Initializing Multi-Agent Testing Suite...")
    app = build_agent_graph()
    
    test_queries = [
        ("SQL Agent", "What tables exist in the database and what are their columns?"),
        ("Math Agent", "Calculate 153 * 42 / 7"),
        ("Python Agent", "Write a python script that prints 'Hello from Python Agent'"),
        ("Web Agent", "What are the latest news headlines today?"),
        ("RAG Agent", "What are my skills and projects?"),
        ("LLM Agent", "Explain the concept of quantum computing in one sentence.")
    ]
    
    success_count = 0
    
    for agent_name, query in test_queries:
        print(f"\n{'='*50}")
        print(f"Testing {agent_name}")
        print(f"Query: '{query}'")
        print(f"{'='*50}")
        
        state = {"messages": [HumanMessage(content=query)]}
        try:
            final_state = app.invoke(state, {"recursion_limit": 15})
            
            route = final_state.get('route')
            draft = final_state.get('draft_response')
            approved = final_state.get('is_approved')
            
            print(f"\n[Result] Routed to: {route}")
            print(f"[Result] Critic Approved: {approved}")
            print(f"[Result] Response:\n{draft}\n")
            
            if approved and draft:
                success_count += 1
                
        except Exception as e:
            print(f"❌ Pipeline Failed for {agent_name}: {str(e)}")
            
    print(f"\n🎯 Test Summary: {success_count}/{len(test_queries)} agents passed successfully.")

if __name__ == "__main__":
    test_pipeline()
