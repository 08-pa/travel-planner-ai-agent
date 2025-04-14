from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict
import json

class DayPlan(BaseModel):
    day_number: int = Field(description="Day number in the itinerary")
    activities: List[str] = Field(description="List of activities for the day")
    meals: Dict[str, str] = Field(description="Meal recommendations")
    accommodation: str = Field(description="Accommodation for the night")

class Itinerary(BaseModel):
    destination: str = Field(description="Name of the destination")
    days: List[DayPlan] = Field(description="Day-by-day plans")
    estimated_budget: str = Field(description="Estimated budget for the trip")

def create_itinerary(state: Dict[str, Any]) -> Dict[str, Any]:
    """Create a detailed day-by-day itinerary using LangChain."""
    
    # Always make sure is_followup is False for the initial run
    state["is_followup"] = False
    
    if not state.get("destinations"):
        state["error"] = "No destinations selected for itinerary creation"
        return state
    
    chat = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")  # Use more widely available model
    
    try:
        # Use the top recommended destination
        destination = state["destinations"][0]
        duration = state["preferences"]["duration"][0]  # Use minimum duration
        
        print(f"Creating itinerary for: {destination['name']}")
        print(f"With parameters: Budget={state['preferences']['budget_level']}, Duration={duration}, "
              f"Interests={state['preferences']['interests']}, Season={state['preferences']['preferred_seasons'][0]}")
        
        # First, create a simpler approach using a normal completion without the complex parser
        prompt = f"""
        Create a detailed travel itinerary for {destination['name']} with the following preferences:
        Budget: {state['preferences']['budget_level']}
        Duration: {duration} days
        Interests: {', '.join(state['preferences']['interests'])}
        Season: {state['preferences']['preferred_seasons'][0]}
        
        Format the response as a JSON object with the following structure:
        {{
          "destination": "{destination['name']}",
          "days": [
            // One object for each day of the trip with:
            {{
              "day_number": 1,
              "activities": ["Activity 1", "Activity 2", "Activity 3"],
              "meals": {{
                "breakfast": "Breakfast venue",
                "lunch": "Lunch venue",
                "dinner": "Dinner venue"
              }},
              "accommodation": "Hotel name or type"
            }}
          ],
          "estimated_budget": "Budget estimate"
        }}
        
        IMPORTANT: Only return valid JSON with no comments or extra text.
        """
        
        response = chat.invoke(prompt)
        print(f"Got response from LLM: {response}")
        
        # Extract the JSON from the response - sometimes the LLM includes extra text
        try:
            # Find the opening brace of the JSON
            start_idx = response.content.find('{')
            end_idx = response.content.rfind('}') + 1
            json_str = response.content[start_idx:end_idx]
            
            # Parse the JSON
            itinerary_data = json.loads(json_str)
            print(f"Parsed itinerary data: {itinerary_data}")
            
            # Add to state
            state["itinerary"] = itinerary_data
            
            # Successfully created the itinerary but don't set is_followup here
            # We'll only set is_followup=True when the user asks a follow-up question
            print("Itinerary created successfully")
            
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from response: {e}")
        
    except Exception as e:
        error_msg = f"Error creating itinerary: {str(e)}"
        print(error_msg)
        state["error"] = error_msg
    
    return state 