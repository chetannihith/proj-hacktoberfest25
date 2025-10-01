import datetime
from typing import Dict, List
import os

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

# Ensure your GROQ_API_KEY is set in your environment variables
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is not set.")

from google.adk.models.lite_llm import LiteLlm
groq_model = LiteLlm(model="groq/gemma2-9b-it")


# Single-file ADK Travel Assistant with multiple tools: Weather + Flights


def get_weather(city: str, day: str = "today") -> Dict[str, str]:
    """Return a mock weather report for a given city and day.

    Args:
        city: city name
        day: one of 'today', 'tomorrow', or a YYYY-MM-DD date

    Returns:
        dict with keys: status, report or error_message
    """
    resolved_day = day
    if day.lower() in {"today", "tomorrow"}:
        base = datetime.date.today()
        delta = 0 if day.lower() == "today" else 1
        resolved_day = str(base + datetime.timedelta(days=delta))

    city_key = city.lower().strip()
    mock = {
        "paris": "Partly cloudy, 22°C high, 12°C low",
        "delhi": "Sunny, 34°C high, 26°C low",
        "bangalore": "Light showers, 26°C high, 19°C low",
        "goa": "Sunny, 30°C high, 25°C low",
    }
    if city_key in mock:
        return {
            "status": "success",
            "report": f"Weather in {city.title()} on {resolved_day}: {mock[city_key]}",
        }
    return {
        "status": "error",
        "error_message": f"Weather for '{city}' not available.",
    }


def find_flight(source_city: str, destination_city: str, day: str = "Friday") -> Dict[str, List[Dict[str, str]]]:
    """Return dummy flights between two cities on a given day.

    Args:
        source_city: departure city
        destination_city: arrival city
        day: day or date descriptor

    Returns:
        dict with status and flights or error_message
    """
    src = source_city.title().strip()
    dst = destination_city.title().strip()
    routes = {
        ("Delhi", "Bangalore"): [
            {"airline": "IndiAir", "dep": "08:15", "arr": "10:45", "price": "₹6,200"},
            {"airline": "AirVista", "dep": "18:30", "arr": "21:00", "price": "₹6,900"},
        ],
        ("Mumbai", "Goa"): [
            {"airline": "IndiAir", "dep": "07:10", "arr": "08:20", "price": "₹3,100"},
            {"airline": "SkyJet", "dep": "20:45", "arr": "21:55", "price": "₹3,600"},
        ],
    }
    flights = routes.get((src, dst))
    if not flights:
        return {
            "status": "error",
            "error_message": f"No dummy flights configured from {src} to {dst}.",
        }
    return {"status": "success", "flights": flights, "day": day}


root_agent = LlmAgent(
    name="travel_assistant",
    model=groq_model,
    description="Helps plan simple travel queries using weather and flight tools.",
    instruction=(
        "You are a travel assistant. Decide which tool(s) to use based on the user's request: "
        "Use get_weather for weather questions; use find_flight for flight search. "
        "For a weekend trip plan, call both tools and then summarize your findings clearly."
    ),
    tools=[get_weather, find_flight],
)


def chat_once(query: str) -> str:
    runner = InMemoryRunner(agent=root_agent)
    events = runner.run(
        user_id="user",
        session_id="session",
        new_message=types.Content(role="user", parts=[types.Part(text=query)]),
    )
    output = []
    for e in events:
        if e.content and e.content.parts:
            for part in e.content.parts:
                if getattr(part, "text", None):
                    output.append(part.text)
    return "\n".join(output).strip()


if __name__ == "__main__":
    while True:
        try:
            q = input("Ask a travel question (e.g., 'Find me a flight from Delhi to Bangalore'): ")
        except EOFError:
            break
        print(chat_once(q))


