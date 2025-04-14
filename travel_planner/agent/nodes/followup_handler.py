from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def handle_followup(state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle follow-up questions about the itinerary."""
    
    print(f"Handling followup question: {state.get('preferences', {}).get('user_input', 'No question found')}")
    
    # Simplified error handling
    if not state.get("itinerary"):
        error_msg = "No itinerary available for follow-up questions"
        print(error_msg)
        state["error"] = error_msg
        return state
    
    if not state.get("preferences") or "user_input" not in state["preferences"]:
        error_msg = "No follow-up question provided"
        print(error_msg)
        state["error"] = error_msg
        return state
    
    try:
        user_question = state["preferences"]["user_input"]
        print(f"Processing follow-up question: {user_question}")
        
        chat = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        
        # Extract key itinerary details for context
        destination = state["itinerary"]["destination"]
        days = len(state["itinerary"]["days"])
        budget = state["itinerary"]["estimated_budget"]
        
        # Collect some activities for context
        activities = []
        for day in state["itinerary"]["days"]:
            activities.extend(day["activities"])
        activities_str = ", ".join(activities[:5])  # Just a few activities for context
        
        prompt = f"""
        You are a helpful travel assistant answering a follow-up question about an itinerary. 
        
        ITINERARY CONTEXT:
        Destination: {destination}
        Duration: {days} days
        Budget: {budget}
        Sample activities: {activities_str}
        
        USER QUESTION: {user_question}
        
        Please provide a helpful and informative answer based on the itinerary details.
        """
        
        response = chat.invoke(prompt)
        answer = response.content
        print(f"Generated answer: {answer[:100]}...")  # Print just the beginning of the answer
        
        # Add response to state
        if "followup_responses" not in state:
            state["followup_responses"] = []
        
        state["followup_responses"].append({
            "question": user_question,
            "answer": answer
        })
        
    except Exception as e:
        error_msg = f"Error handling follow-up question: {str(e)}"
        print(error_msg)
        state["error"] = error_msg
    
    return state 