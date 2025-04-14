import json
from typing import Dict, Any, List
from pathlib import Path

def load_destinations() -> List[Dict[str, Any]]:
    """Load destinations from the JSON database."""
    try:
        db_path = Path(__file__).parent.parent / "data" / "destinations.json"
        with open(db_path, "r") as f:
            data = json.load(f)
            print(f"Loaded destinations: {data}")  # Debug print
            return data["destinations"]
    except Exception as e:
        print(f"Error loading destinations: {e}")  # Debug print
        raise

def match_destinations(preferences: Dict[str, Any], destinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Match destinations based on user preferences."""
    print(f"Matching with preferences: {preferences}")  # Debug print
    matched = []
    
    for dest in destinations:
        score = 0
        print(f"Checking destination: {dest}")  # Debug print
        
        # Match budget level
        if dest["budget_level"] == preferences.get("budget_level", "").lower():
            score += 2
            print(f"Budget matched: +2")  # Debug print
            
        # Match duration
        user_min, user_max = preferences.get("duration", [0, 0])
        dest_min, dest_max = dest["ideal_duration"]
        if (dest_min <= user_max and dest_max >= user_min):
            score += 1
            print(f"Duration matched: +1")  # Debug print
            
        # Match interests with tags
        user_interests = set(preferences.get("interests", []))
        dest_tags = set(dest["tags"])
        matching_interests = user_interests.intersection(dest_tags)
        score += len(matching_interests)
        print(f"Matching interests: {matching_interests}, score: +{len(matching_interests)}")  # Debug print
        
        # Match seasons
        user_seasons = set(preferences.get("preferred_seasons", []))
        dest_seasons = set(dest["best_seasons"])
        matching_seasons = user_seasons.intersection(dest_seasons)
        if matching_seasons:
            score += 1
            print(f"Seasons matched: +1")  # Debug print
            
        print(f"Final score for {dest['name']}: {score}")  # Debug print
        
        if score > 0:  # Lower the threshold to see more matches
            dest["match_score"] = score
            matched.append(dest)
    
    return sorted(matched, key=lambda x: x["match_score"], reverse=True)

def find_destinations(state: Dict[str, Any]) -> Dict[str, Any]:
    """Find suitable destinations based on user preferences."""
    try:
        print(f"Initial state: {state}")  # Debug print
        destinations = load_destinations()
        matched_destinations = match_destinations(state["preferences"], destinations)
        print(f"Matched destinations: {matched_destinations}")  # Debug print
        state["destinations"] = matched_destinations[:3]  # Top 3 matches
    except Exception as e:
        print(f"Error in find_destinations: {e}")  # Debug print
        state["error"] = f"Error finding destinations: {str(e)}"
    
    return state 