import streamlit as st
from typing import Dict, List, Any, TypedDict, Optional
from dotenv import load_dotenv
import os
import time
import json
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

# Import our nodes
from nodes.preference_extractor import extract_preferences
from nodes.destination_finder import find_destinations
from nodes.itinerary_creator import create_itinerary
from nodes.followup_handler import handle_followup

# Define state schema
class TravelPlannerState(TypedDict):
    preferences: Dict[str, Any]
    destinations: List[Dict[str, Any]]
    itinerary: Dict[str, Any]
    history: List[Dict[str, str]]
    is_followup: bool
    error: Optional[str]
    followup_responses: List[Dict[str, str]]

# Load environment variables
load_dotenv()

# Check if API key is available
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

# Build the travel agent graph
def build_travel_agent():
    workflow = StateGraph(TravelPlannerState)
    
    # Add nodes
    workflow.add_node("extract_preferences", extract_preferences)
    workflow.add_node("find_destinations", find_destinations)
    workflow.add_node("create_itinerary", create_itinerary)
    
    # Add end node
    workflow.add_node("end", lambda x: x)
    
    # Add edges
    workflow.add_edge("extract_preferences", "find_destinations")
    workflow.add_edge("find_destinations", "create_itinerary")
    workflow.add_edge("create_itinerary", "end")
    
    # Set entry point
    workflow.set_entry_point("extract_preferences")
    
    return workflow.compile()

# Build the followup agent
def build_followup_agent():
    workflow = StateGraph(TravelPlannerState)
    
    # Add the followup handler
    workflow.add_node("handle_followup", handle_followup)
    
    # Add end node
    workflow.add_node("end", lambda x: x)
    
    # Add edge
    workflow.add_edge("handle_followup", "end")
    
    # Set entry point
    workflow.set_entry_point("handle_followup")
    
    return workflow.compile()

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.preferences_submitted = False
    st.session_state.itinerary_generated = False
    st.session_state.destinations = []
    st.session_state.itinerary = {}
    st.session_state.chat_history = []

# Page Configuration
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a more beautiful interface
st.markdown("""
<style>
    body {
        background-color: #121212;
        color: white;
    }
    .main-header {
        font-size: 2.5rem;
        color: #90CAF9;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #90CAF9;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        color: white;
    }
    .destination-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        border: 1px solid #444;
        color: white;
    }
    .day-card {
        background-color: #242424;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #90CAF9;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        color: white;
    }
    .chat-message {
        margin-bottom: 10px;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }
    .user-message {
        background-color: #2C3E50;
        border-left: 5px solid #3498DB;
        color: white;
    }
    .ai-message {
        background-color: #1A472A;
        border-left: 5px solid #2ECC71;
        color: white;
    }
    .info-text {
        color: #CCCCCC;
        font-size: 0.9rem;
    }
    .highlight {
        color: #90CAF9;
        font-weight: bold;
    }
    /* Improved tabs visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #333333;
        border-radius: 4px 4px 0px 0px;
        padding: 8px 16px;
        border: 1px solid #444;
        border-bottom: none;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
        font-weight: bold;
    }
    h3, h4, p, ul, li {
        color: white;
    }
    .stApp {
        background-color: #121212;
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("<h1 style='font-size: 2.5rem; color: #90CAF9; text-align: center; margin-bottom: 1rem;'>✈️ AI Travel Planner</h1>", unsafe_allow_html=True)
st.markdown("""
<div style="background-color: #1E1E1E; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); color: white;">
    <p>Welcome to your personal AI Travel Planner! Tell us about your travel preferences, 
    and our AI will suggest destinations, create a detailed itinerary, and answer your travel questions.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1488646953014-85cb44e25828?q=80&w=300", width=300)
    st.markdown("<h2 class='sub-header'>How it works</h2>", unsafe_allow_html=True)
    st.markdown("""
    1. **Share your preferences** - Tell us about your budget, trip duration, interests, and preferred seasons.
    2. **Get recommendations** - Our AI will suggest destinations that match your preferences.
    3. **View your itinerary** - See a detailed day-by-day plan for your selected destination.
    4. **Ask questions** - Have follow-up questions? Our AI will provide the answers.
    """)
    
    if st.session_state.itinerary_generated:
        st.success("Your travel plan is ready! 🎉")
    elif st.session_state.preferences_submitted:
        st.info("Processing your preferences... ⏳")
    else:
        st.warning("Please share your travel preferences to get started.")

# Main app logic
if not st.session_state.preferences_submitted:
    # Travel Preferences Form
    st.markdown("<h2 class='sub-header'>Your Travel Preferences</h2>", unsafe_allow_html=True)
    
    with st.form(key='preferences_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            budget_level = st.selectbox(
                "Budget Level",
                options=["low", "medium", "high"],
                format_func=lambda x: {"low": "Budget-friendly", "medium": "Mid-range", "high": "Luxury"}[x]
            )
            
            # Season selection with multiselect
            seasons = st.multiselect(
                "Preferred Seasons",
                options=["spring", "summer", "fall", "winter"],
                default=["summer"]
            )
        
        with col2:
            # Duration range with slider
            min_days, max_days = st.slider(
                "Trip Duration (days)",
                min_value=1,
                max_value=30,
                value=(5, 10)
            )
            
            # Interests with multiselect
            interests = st.multiselect(
                "Travel Interests",
                options=["beach", "culture", "food", "history", "nature", "architecture", 
                         "photography", "adventure", "relaxation", "shopping", "nightlife"],
                default=["culture", "food"]
            )
        
        additional_info = st.text_area(
            "Anything else you'd like to add?",
            placeholder="E.g., looking for family-friendly activities, prefer walkable cities, etc."
        )
        
        submit_button = st.form_submit_button(label="Plan My Trip", use_container_width=True)
        
        if submit_button:
            # Create user input string
            user_input = f"""
            I'm looking for a {min_days}-{max_days} day trip with a {budget_level} budget.
            I'm interested in {', '.join(interests)}.
            I prefer traveling in {', '.join(seasons)}.
            Additional information: {additional_info}
            """
            
            # Show spinner during processing
            with st.spinner("Planning your perfect trip..."):
                try:
                    # Initialize agent
                    agent = build_travel_agent()
                    
                    # Create initial state
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
                    final_state = agent.invoke(initial_state)
                    
                    if final_state.get("error"):
                        st.error(f"Error: {final_state['error']}")
                    else:
                        # Store the results in session state
                        st.session_state.destinations = final_state.get("destinations", [])
                        st.session_state.itinerary = final_state.get("itinerary", {})
                        st.session_state.preferences_submitted = True
                        
                        if st.session_state.itinerary:
                            st.session_state.itinerary_generated = True
                        
                        # Rerun to update the UI
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

elif st.session_state.preferences_submitted:
    # Add a restart button
    if st.button("Start Over", type="secondary"):
        st.session_state.preferences_submitted = False
        st.session_state.itinerary_generated = False
        st.session_state.destinations = []
        st.session_state.itinerary = {}
        st.session_state.chat_history = []
        st.rerun()
    
    # Display Destination Recommendations
    if st.session_state.destinations:
        st.markdown("<h2 style='font-size: 1.5rem; color: #90CAF9; margin-top: 1.5rem; margin-bottom: 1rem;'>Recommended Destinations</h2>", unsafe_allow_html=True)
        
        # Create a row of destination cards
        cols = st.columns(min(3, len(st.session_state.destinations)))
        
        for i, dest in enumerate(st.session_state.destinations):
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div style="background-color: #1E1E1E; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3); border: 1px solid #444; color: white;">
                    <h3 style="color: white;">{dest['name']}, {dest['country']}</h3>
                    <p><span style="color: #90CAF9; font-weight: bold;">Budget:</span> {dest['budget_level'].capitalize()}</p>
                    <p><span style="color: #90CAF9; font-weight: bold;">Best For:</span> {', '.join(dest['tags'])}</p>
                    <p><span style="color: #90CAF9; font-weight: bold;">Ideal Duration:</span> {dest['ideal_duration'][0]}-{dest['ideal_duration'][1]} days</p>
                    <p><span style="color: #90CAF9; font-weight: bold;">Best Seasons:</span> {', '.join(dest['best_seasons']).capitalize()}</p>
                    <p><span style="color: #90CAF9; font-weight: bold;">Match Score:</span> {dest.get('match_score', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Display Itinerary
    if st.session_state.itinerary:
        st.markdown("<h2 style='font-size: 1.5rem; color: #90CAF9; margin-top: 1.5rem; margin-bottom: 1rem;'>Your Travel Itinerary</h2>", unsafe_allow_html=True)
        
        itinerary = st.session_state.itinerary
        
        # Itinerary header
        st.markdown(f"""
        <div style="background-color: #1E1E1E; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); color: white;">
            <h3 style="color: white;">🌍 {itinerary['destination']}</h3>
            <p><span style="color: #90CAF9; font-weight: bold;">Duration:</span> {len(itinerary['days'])} days</p>
            <p><span style="color: #90CAF9; font-weight: bold;">Estimated Budget:</span> {itinerary['estimated_budget']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tabs for Days
        tabs = st.tabs([f"Day {day['day_number']}" for day in itinerary['days']])
        
        for i, day in enumerate(itinerary['days']):
            with tabs[i]:
                # Render each part separately with inline styles
                st.markdown(f"""<div style="background-color: #242424; border-radius: 8px; padding: 15px; margin-bottom: 10px; border-left: 5px solid #90CAF9; color: white;">""", unsafe_allow_html=True)
                
                st.markdown(f"<h3 style='color: white;'>Day {day['day_number']}</h3>", unsafe_allow_html=True)
                
                st.markdown("<h4 style='color: #90CAF9;'>🏞️ Activities</h4>", unsafe_allow_html=True)
                activities_list = "".join([f"<li>{activity}</li>" for activity in day['activities']])
                st.markdown(f"<ul style='color: white;'>{activities_list}</ul>", unsafe_allow_html=True)
                
                st.markdown("<h4 style='color: #90CAF9;'>🍽️ Meals</h4>", unsafe_allow_html=True)
                st.markdown(f"<p><span style='color: #90CAF9; font-weight: bold;'>Breakfast:</span> {day['meals'].get('breakfast', 'Not specified')}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><span style='color: #90CAF9; font-weight: bold;'>Lunch:</span> {day['meals'].get('lunch', 'Not specified')}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><span style='color: #90CAF9; font-weight: bold;'>Dinner:</span> {day['meals'].get('dinner', 'Not specified')}</p>", unsafe_allow_html=True)
                
                st.markdown("<h4 style='color: #90CAF9;'>🏨 Accommodation</h4>", unsafe_allow_html=True)
                st.markdown(f"<p>{day['accommodation']}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True) # Close the div
        
        # Follow-up Questions Chat Interface
        st.markdown("<h2 style='font-size: 1.5rem; color: #90CAF9; margin-top: 1.5rem; margin-bottom: 1rem;'>Questions About Your Trip</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #CCCCCC; font-size: 0.9rem;'>Ask any questions about your itinerary, local attractions, or travel tips.</p>", unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""<div style="margin-bottom: 10px; padding: 10px; border-radius: 8px; background-color: #2C3E50; border-left: 5px solid #3498DB; color: white; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);">
                                <b>You:</b> {message["content"]}
                             </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div style="margin-bottom: 10px; padding: 10px; border-radius: 8px; background-color: #1A472A; border-left: 5px solid #2ECC71; color: white; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);">
                                <b>AI Travel Assistant:</b> {message["content"]}
                             </div>""", unsafe_allow_html=True)
        
        # Chat input
        user_question = st.chat_input("Ask a question about your trip...")
        
        if user_question:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Display the user message immediately
            st.markdown(f"""<div style="margin-bottom: 10px; padding: 10px; border-radius: 8px; background-color: #2C3E50; border-left: 5px solid #3498DB; color: white; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);">
                            <b>You:</b> {user_question}
                         </div>""", unsafe_allow_html=True)
            
            # Process the followup question
            try:
                followup_agent = build_followup_agent()
                
                followup_state: TravelPlannerState = {
                    "preferences": {"user_input": user_question},
                    "itinerary": st.session_state.itinerary,
                    "is_followup": True,
                    "destinations": st.session_state.destinations,
                    "history": [],
                    "error": None,
                    "followup_responses": []
                }
                
                # Display a spinner while processing
                with st.spinner("Thinking..."):
                    followup_result = followup_agent.invoke(followup_state)
                
                if followup_result.get("error"):
                    answer = f"Sorry, I couldn't process your question: {followup_result['error']}"
                elif followup_result.get("followup_responses"):
                    answer = followup_result["followup_responses"][-1]["answer"]
                else:
                    answer = "I'm sorry, I couldn't generate an answer to your question. Please try asking something else about your itinerary."
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
                # Display the assistant response
                st.markdown(f"""<div style="margin-bottom: 10px; padding: 10px; border-radius: 8px; background-color: #1A472A; border-left: 5px solid #2ECC71; color: white; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);">
                                <b>AI Travel Assistant:</b> {answer}
                             </div>""", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred while processing your question: {str(e)}")
    
    else:
        st.warning("No itinerary was generated. This could be due to no matching destinations or an error in the planning process. Please try again with different preferences.") 