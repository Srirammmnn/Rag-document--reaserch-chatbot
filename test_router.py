import os
from dotenv import load_dotenv
load_dotenv()

from agent import build_agent_graph
from langchain_core.messages import HumanMessage

def test_full_pipeline():
    print("Testing Multi-Agent Graph Pipeline...\n")
    app = build_agent_graph()
    
    # Test SQL Agent
    question = "What was the total revenue in 2023?"
    print(f"User Query: '{question}'")
    
    state = {"messages": [HumanMessage(content=question)]}
    
    try:
        final_state = app.invoke(state, {"recursion_limit": 10})
        
        print("\n=== FINAL STATE ===")
        print(f"Route Taken: {final_state.get('route')}")
        print(f"Critic Approved: {final_state.get('is_approved')}")
        print(f"Final Draft Response:\n{final_state.get('draft_response')}")
        
    except Exception as e:
        print(f"\nPipeline Failed: {e}")

if __name__ == "__main__":
    test_full_pipeline()
