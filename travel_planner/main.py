from typing import Dict, List, Any, TypedDict, Optional
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from nodes.preference_extractor import extract_preferences
from nodes.destination_finder import find_destinations
from nodes.itinerary_creator import create_itinerary
from nodes.followup_handler import handle_followup
from state import AgentState
from dotenv import load_dotenv
import os

# Define a proper state schema
class TravelPlannerState(TypedDict):
    preferences: Dict[str, Any]
    destinations: List[Dict[str, Any]]
    itinerary: Dict[str, Any]
    history: List[Dict[str, str]]
    is_followup: bool
    error: Optional[str]
    followup_responses: List[Dict[str, str]]

# Load environment variables from .env file
load_dotenv()

# Verify OpenAI API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
else:
    print(f"Loaded API key: {api_key[:10]}...")

def get_user_input():
    print("\nWelcome to the Travel Planner AI!")
    print("\nPlease tell me about your travel preferences.")
    print("Include information about:")
    print("- Your budget (low/medium/high)")
    print("- Desired trip duration (in days)")
    print("- Your interests/activities")
    print("- Preferred seasons to travel\n")
    return input("Your preferences: ")

def should_go_to_followup(state):
    """Conditional router to determine if we should go to followup node."""
    print(f"Checking if we should go to followup: is_followup={state.get('is_followup', False)}")
    return "handle_followup" if state.get("is_followup", False) else "end"

def build_travel_agent():
    # Initialize the state graph with the proper typed state
    workflow = StateGraph(TravelPlannerState)
    
    # Add nodes
    workflow.add_node("extract_preferences", extract_preferences)
    workflow.add_node("find_destinations", find_destinations)
    workflow.add_node("create_itinerary", create_itinerary)
    workflow.add_node("handle_followup", handle_followup)
    
    # Add end node (returns the final state)
    workflow.add_node("end", lambda x: x)
    
    # Add edges
    workflow.add_edge("extract_preferences", "find_destinations")
    workflow.add_edge("find_destinations", "create_itinerary")
    workflow.add_conditional_edges(
        "create_itinerary",
        should_go_to_followup
    )
    workflow.add_edge("handle_followup", "end")
    
    # Set entry point
    workflow.set_entry_point("extract_preferences")
    
    return workflow.compile()

def build_followup_agent():
    """Build a simplified agent just for handling followup questions."""
    workflow = StateGraph(TravelPlannerState)
    
    # Just the followup handler
    workflow.add_node("handle_followup", handle_followup)
    
    # Add end node
    workflow.add_node("end", lambda x: x)
    
    # Add edge
    workflow.add_edge("handle_followup", "end")
    
    # Set entry point
    workflow.set_entry_point("handle_followup")
    
    return workflow.compile()

if __name__ == "__main__":
    # Build the main travel planning agent
    agent = build_travel_agent()
    
    # Get user input
    user_input = get_user_input()
    
    # Initialize with user input
    initial_state: TravelPlannerState = {
        "preferences": {"user_input": user_input},
        "destinations": [],
        "itinerary": {},
        "history": [],
        "is_followup": False,
        "error": None,
        "followup_responses": []
    }
    
    # Run the agent
    print(f"Starting agent with initial state: {initial_state}")
    final_state = agent.invoke(initial_state)
    print(f"Final state after initial run: {final_state}")
    
    # Display results
    if final_state.get("error"):
        print(f"\nError: {final_state['error']}")
    else:
        print("\nRecommended Destinations:")
        for dest in final_state.get("destinations", []):
            print(f"- {dest['name']}, {dest['country']}")
        
        if final_state.get("itinerary"):
            print("\nGenerated Itinerary:")
            print(f"Destination: {final_state['itinerary']['destination']}")
            for day in final_state['itinerary']['days']:
                print(f"\nDay {day['day_number']}:")
                print("Activities:", ", ".join(day['activities']))
                print("Meals:", day['meals'])
                print("Accommodation:", day['accommodation'])
            
            print(f"\nEstimated Budget: {final_state['itinerary'].get('estimated_budget', 'Not available')}")
            
            # Create a separate agent for follow-up questions
            followup_agent = build_followup_agent()
            
            # Ask for follow-up questions
            while True:
                print("\nDo you have any questions about your itinerary? (or type 'exit' to end)")
                question = input("Your question: ")
                if question.lower() == 'exit':
                    break
                
                # Create a new state for the follow-up that's simpler
                followup_state: TravelPlannerState = {
                    "preferences": {"user_input": question},
                    "itinerary": final_state.get("itinerary", {}),
                    "is_followup": True,  # This is a follow-up question
                    "destinations": [],
                    "history": [],
                    "error": None,
                    "followup_responses": []
                }
                
                # Run the followup agent directly on the followup state
                followup_result = followup_agent.invoke(followup_state)
                
                if followup_result.get("error"):
                    print(f"\nError: {followup_result['error']}")
                elif followup_result.get("followup_responses"):
                    print("\nAnswer:", followup_result["followup_responses"][-1]["answer"])
                else:
                    print("\nNo answer was generated. Please try a different question.")
        else:
            print("\nNo itinerary was generated. Please try again with different preferences.") 