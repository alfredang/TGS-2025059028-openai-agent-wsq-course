from dotenv import load_dotenv
import os
import streamlit as st
import asyncio
from pydantic import BaseModel
from agents import Runner, Agent, function_tool
from tavily import TavilyClient

# run this script with `streamlit run app/trip_planner.py`

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
tavily_key = os.environ.get("TAVILY_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

if not tavily_key:
    raise ValueError("TAVILY_API_KEY is not set in the environment variables")


# ---------- TAVILY SEARCH TOOLS ----------

@function_tool
def search_attractions(query: str) -> str:
    """
    Search for tourist attractions, landmarks, and activities at a destination.
    Use this to find current information about places to visit, opening hours, and popular activities.
    """
    tavily = TavilyClient(api_key=tavily_key)
    response = tavily.search(query=f"{query} tourist attractions things to do", search_depth="advanced")
    results = response.get('results', [])
    summary = "\n\n".join([f"Source: {res['url']}\nContent: {res['content']}" for res in results[:5]])
    return summary if summary else "No results found."


@function_tool
def search_accommodation_prices(query: str) -> str:
    """
    Search for accommodation prices, hotel costs, and lodging options at a destination.
    Use this to find current pricing for hotels, hostels, and other accommodations.
    """
    tavily = TavilyClient(api_key=tavily_key)
    response = tavily.search(query=f"{query} hotel prices accommodation cost per night", search_depth="advanced")
    results = response.get('results', [])
    summary = "\n\n".join([f"Source: {res['url']}\nContent: {res['content']}" for res in results[:5]])
    return summary if summary else "No results found."


@function_tool
def search_food_restaurants(query: str) -> str:
    """
    Search for local food, restaurants, and dining recommendations at a destination.
    Use this to find popular restaurants, local cuisine, and food prices.
    """
    tavily = TavilyClient(api_key=tavily_key)
    response = tavily.search(query=f"{query} best restaurants local food must try dishes", search_depth="advanced")
    results = response.get('results', [])
    summary = "\n\n".join([f"Source: {res['url']}\nContent: {res['content']}" for res in results[:5]])
    return summary if summary else "No results found."


@function_tool
def search_transport_costs(query: str) -> str:
    """
    Search for transportation costs and options at a destination.
    Use this to find information about public transport, taxis, and getting around.
    """
    tavily = TavilyClient(api_key=tavily_key)
    response = tavily.search(query=f"{query} transportation costs public transport prices getting around", search_depth="advanced")
    results = response.get('results', [])
    summary = "\n\n".join([f"Source: {res['url']}\nContent: {res['content']}" for res in results[:5]])
    return summary if summary else "No results found."


@function_tool
def search_local_tips(query: str) -> str:
    """
    Search for local tips, cultural etiquette, and travel advice for a destination.
    Use this to find insider tips, customs, and practical travel advice.
    """
    tavily = TavilyClient(api_key=tavily_key)
    response = tavily.search(query=f"{query} travel tips local customs etiquette advice", search_depth="advanced")
    results = response.get('results', [])
    summary = "\n\n".join([f"Source: {res['url']}\nContent: {res['content']}" for res in results[:5]])
    return summary if summary else "No results found."


# ---------- OUTPUT SCHEMA ----------

class TravelOutput(BaseModel):
    destination: str
    duration: str
    summary: str
    cost: str
    tips: str


# ---------- SPECIALIZED AGENTS (Used as Tools via .as_tool()) ----------

planner_agent = Agent(
    name="Planner Agent",
    model="gpt-4.1-mini",
    handoff_description="Use me when the user asks to plan or outline an itinerary, schedule, or daily plan.",
    instructions=(
        "You specialize in building day-by-day travel itineraries and sequencing activities. "
        "ALWAYS use the search_attractions tool to find current, accurate information about attractions and activities. "
        "Focus on:\n"
        "- Popular attractions and must-see locations (search for them!)\n"
        "- Optimal timing and logical pacing\n"
        "- Mix of activities (cultural, recreational, relaxation)\n"
        "- Practical logistics (travel time between locations)\n\n"
        "Search for real information before creating the itinerary. "
        'Return a detailed day-by-day plan with specific times, attractions, and activities.'
    ),
    tools=[search_attractions]
)

budget_agent = Agent(
    name="Budget Agent",
    model="gpt-4.1-mini",
    handoff_description="Use me when the user mentions budget, price, cost, dollars, under $X, or asks 'how much'.",
    instructions=(
        "You estimate costs for lodging, food, transport, and activities. "
        "ALWAYS use the search tools to find current, accurate pricing information:\n"
        "- Use search_accommodation_prices for hotel/hostel costs\n"
        "- Use search_transport_costs for transportation costs\n\n"
        "Include estimates for:\n"
        "- Accommodation (hotels/hostels) - SEARCH for real prices\n"
        "- Food (meals per day)\n"
        "- Transportation (local + intercity) - SEARCH for real costs\n"
        "- Activities and entrance fees\n"
        "- Miscellaneous expenses\n\n"
        "Flag if budget might be exceeded. Provide specific price ranges in USD based on your research."
    ),
    tools=[search_accommodation_prices, search_transport_costs]
)

local_guide_agent = Agent(
    name="Local Guide Agent",
    model="gpt-4.1-mini",
    handoff_description="Use me when the user asks for food, restaurants, neighborhoods, local tips, or 'what's good nearby'.",
    instructions=(
        "You provide restaurants, neighborhoods, cultural tips, and current local highlights. "
        "ALWAYS use the search tools to find accurate, current information:\n"
        "- Use search_food_restaurants for restaurant and food recommendations\n"
        "- Use search_local_tips for cultural tips and travel advice\n\n"
        "Include:\n"
        "- Must-try local dishes and specific restaurant names (SEARCH for them!)\n"
        "- Hidden gems and off-the-beaten-path spots\n"
        "- Cultural tips and etiquette (SEARCH for current customs)\n"
        "- Practical advice (transport cards, best times to visit, etc.)\n\n"
        "Be specific with restaurant names and locations based on your research."
    ),
    tools=[search_food_restaurants, search_local_tips]
)


# ---------- TRAVEL AGENT (Orchestrator using .as_tool()) ----------

travel_agent = Agent(
    name="Travel Agent",
    model="gpt-4.1-mini",
    instructions=(
        "You are a friendly and knowledgeable travel planner that helps users plan trips, "
        "suggest destinations, and create detailed summaries of their journeys.\n"
        "Your primary role is to orchestrate other specialized agents (used as tools) to complete the user's request.\n"
        "\n"
        "IMPORTANT: Each specialized agent has Tavily search tools to find accurate, real-time information. "
        "Make sure to tell them to use their search tools for accurate data.\n"
        "\n"
        "When planning an itinerary, call the **Planner Agent** to create daily schedules, "
        "organize destinations, and recommend attractions or activities. Do not create itineraries yourself.\n"
        "When estimating costs, call the **Budget Agent** to calculate the total trip cost "
        "including flights, hotels, and activities. Do not calculate or estimate prices on your own.\n"
        "When recommending local experiences, restaurants, neighborhoods, or cultural highlights, "
        "call the **Local Guide Agent** to provide these insights. Do not generate local recommendations without this agent.\n"
        "\n"
        "Use these agents one at a time in a logical order based on the request — "
        "start with the Planner Agent, then the Budget Agent, and finally the Local Guide Agent.\n"
        "After receiving results from these agents, combine their outputs into a single structured summary.\n"
        "\n"
        "Return JSON output using this exact structure:\n"
        '{"destination": "string", "duration": "string", "summary": "string", "cost": "string", "tips": "string"}.\n'
    ),
    output_type=TravelOutput,
    tools=[
        planner_agent.as_tool(
            tool_name="planner_agent",
            tool_description="Plan or outline an itinerary, schedule, or daily plan for the trip. This agent searches for real attractions and activities."
        ),
        budget_agent.as_tool(
            tool_name="budget_agent",
            tool_description="Calculate and estimate the cost of a trip including accommodation, food, and activities. This agent searches for real prices."
        ),
        local_guide_agent.as_tool(
            tool_name="local_guide_agent",
            tool_description="Provide restaurants, neighborhoods, cultural tips, and current local highlights. This agent searches for real recommendations."
        )
    ]
)


# ---------- STREAMLIT UI ----------

st.set_page_config(
    page_title="AI Trip Planner",
    page_icon="🧳",
    layout="wide"
)

st.title("🧳 Multi-Agent AI Trip Planner")
st.markdown("""
This app uses a **Travel Agent** that orchestrates specialized AI agents **as tools** using `.as_tool()`:

| Agent | Role | Search Tools |
|-------|------|--------------|
| ✈️ **Travel Agent** | Orchestrator | Coordinates all agents |
| 🧠 **Planner Agent** | Itinerary | `search_attractions` |
| 💰 **Budget Agent** | Costs | `search_accommodation_prices`, `search_transport_costs` |
| 🍣 **Local Guide Agent** | Tips | `search_food_restaurants`, `search_local_tips` |

Each agent uses **Tavily Search** for accurate, real-time information!
""")

st.divider()

# User Inputs
col1, col2 = st.columns(2)

with col1:
    destination = st.text_input("🌍 Destination", value="Tokyo", help="Enter your travel destination")
    days = st.number_input("📅 Number of Days", min_value=1, max_value=14, value=5, help="How many days is your trip?")

with col2:
    budget = st.number_input("💵 Budget (USD)", min_value=500, max_value=20000, value=2000, step=100, help="Your total trip budget")
    st.metric("Daily Budget", f"${budget/days:.2f}")

preferences = st.text_area(
    "✨ Special Preferences (optional)",
    placeholder="e.g., vegetarian food, art museums, nightlife, family-friendly, adventure activities...",
    help="Tell us about your interests and any special requirements"
)

st.divider()

# Plan Trip Button
if st.button("🚀 Plan My Trip", type="primary", use_container_width=True):
    with st.spinner("🤖 Travel Agent is coordinating with specialized agents (searching for real-time info)..."):
        # Create the user request
        user_request = f"""Plan a {days}-day trip to {destination} with a budget of ${budget}.
        Special preferences: {preferences if preferences else 'None specified'}

        Please use all your agent tools to create a comprehensive travel plan:
        1. Use planner_agent to create a day-by-day itinerary (it will search for real attractions)
        2. Use budget_agent to estimate costs and ensure we stay under ${budget} (it will search for real prices)
        3. Use local_guide_agent to get food recommendations and local tips (it will search for real restaurants)

        Combine all the information into a complete travel plan with accurate, researched information."""

        # Run the Travel Agent (which will use other agents as tools)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                Runner.run(travel_agent, input=user_request)
            )

            # Store structured results in session state
            st.session_state.trip_result = result.final_output
            st.session_state.trip_destination = destination
            st.session_state.trip_days = days
            st.session_state.trip_budget = budget

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            loop.close()

# Display Results
if "trip_result" in st.session_state:
    result = st.session_state.trip_result

    st.success(f"✅ Your {st.session_state.trip_days}-day trip to {st.session_state.trip_destination} is ready!")

    # Display structured output if available
    if isinstance(result, TravelOutput):
        # Create tabs for organized display
        tab1, tab2, tab3 = st.tabs(["📅 Itinerary", "💰 Budget", "🍣 Local Tips"])

        with tab1:
            st.header(f"📅 {result.destination} - {result.duration}")
            st.markdown(result.summary)

        with tab2:
            st.header("💰 Budget Breakdown")
            st.markdown(result.cost)

        with tab3:
            st.header("🍣 Local Food & Tips")
            st.markdown(result.tips)
    else:
        # Fallback for string output
        st.header("🗺️ Your Complete Travel Plan")
        st.markdown(str(result))

    # Option to clear and start over
    if st.button("🔄 Plan Another Trip"):
        del st.session_state.trip_result
        del st.session_state.trip_destination
        del st.session_state.trip_days
        del st.session_state.trip_budget
        st.rerun()

# Footer
st.divider()
st.caption("Built with Streamlit, OpenAI Agents SDK & Tavily Search | Agent-as-Tools Pattern")
