from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

class TravelPreferences(BaseModel):
    budget_level: str = Field(description="Budget level (low/medium/high)")
    duration: List[int] = Field(description="Preferred trip duration range in days [min, max]")
    interests: List[str] = Field(description="List of travel interests/activities")
    preferred_seasons: List[str] = Field(description="Preferred seasons for travel")

def extract_preferences(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract travel preferences from user input using LangChain."""
    
    if "user_input" not in state.get("preferences", {}):
        state["error"] = "No user input provided"
        return state
    
    # Initialize ChatOpenAI
    chat = ChatOpenAI(temperature=0)
    
    # Create output parser
    parser = PydanticOutputParser(pydantic_object=TravelPreferences)
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a travel planning assistant. Extract travel preferences from the user's input.
        Format them into:
        - budget_level: must be exactly 'low', 'medium', or 'high'
        - duration: [min_days, max_days] as integers
        - interests: list of activities/interests as lowercase strings
        - preferred_seasons: list of seasons as lowercase strings (spring, summer, fall, winter)
        
        Example output:
        {{
            "budget_level": "medium",
            "duration": [5, 7],
            "interests": ["culture", "food", "history", "architecture"],
            "preferred_seasons": ["spring", "fall"]
        }}"""),
        ("user", "{user_input}")
    ])
    
    # Create chain
    chain = prompt | chat | parser
    
    # Extract preferences
    try:
        print(f"Processing user input: {state['preferences']['user_input']}")  # Debug print
        preferences = chain.invoke({"user_input": state["preferences"]["user_input"]})
        preferences_dict = preferences.dict()
        print(f"Extracted preferences: {preferences_dict}")  # Debug print
        
        # Ensure budget level is lowercase
        preferences_dict["budget_level"] = preferences_dict["budget_level"].lower()
        
        # Update state with formatted preferences
        state["preferences"] = preferences_dict
        state["is_followup"] = False
        print(f"Updated state preferences: {state['preferences']}")  # Debug print
    except Exception as e:
        print(f"Error in preference extraction: {e}")  # Debug print
        state["error"] = f"Error extracting preferences: {str(e)}"
    
    return state 