# Travel Planner AI Agent

A LangGraph-based AI travel planning assistant that helps users plan their trips by understanding preferences, suggesting destinations, creating itineraries, and handling follow-up questions.

## Features

- Extract travel preferences from user input
- Recommend destinations based on preferences
- Generate detailed day-by-day itineraries
- Handle follow-up questions about the travel plan
- State management for conversation history

## Project Structure

```
travel_planner/
├── data/
│   └── destinations.json    # Mock destination database
├── nodes/
│   ├── preference_extractor.py
│   ├── destination_finder.py
│   ├── itinerary_creator.py
│   └── followup_handler.py
├── main.py                 # Entry point
├── state.py                 # State management
├── tests/
│   └── streamlit_app.py # Visual test interface ← *Creative Addition*              
└── requirements.txt       # Dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the travel planner:
```bash
python main.py
```


The agent will:
1. Extract your travel preferences
2. Suggest suitable destinations
3. Create a detailed itinerary
4. Answer follow-up questions about the plan

## Example Interaction

```python
# Initialize the agent
agent = build_travel_agent()

# Start with user preferences
initial_state = AgentState(
    preferences={},
    destinations=[],
    itinerary={},
    history=[]
)

# Run the agent
final_state = agent.invoke(initial_state)
```


#Creative Enhancements
### Streamlit Testing Dashboard (`streamlit_app.py`)

## Technical Implementation

- Uses LangGraph for workflow management
- Implements a multi-step reasoning workflow
- Includes state management and error handling
- Uses LangChain for natural language processing
- Structured data handling with Pydantic models

## Error Handling

The agent includes comprehensive error handling:
- Input validation
- API error handling
- State management error handling
- Graceful fallbacks for each step 



