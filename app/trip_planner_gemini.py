from dotenv import load_dotenv
import os
import streamlit as st
import asyncio
import re
from agents import Runner, Agent, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool
from tavily import TavilyClient

set_tracing_disabled(disabled=True)

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

# ---- Streamlit Page Config ----
st.set_page_config(
    page_title="AI Trip Planner",
    page_icon="🧳",
    layout="wide"
)

st.title("🧳 AI Trip Planner (Multi-Agent System)")
st.markdown("Plan your perfect trip with our team of specialized AI agents!")

client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ---- Tavily Search Tool (limited usage) ----
@function_tool
def search_web(query: str) -> str:
    """Search the web for current travel information. Use sparingly - only for essential real-time info like current events, recent openings, or travel advisories."""
    if not tavily_client:
        return "Web search unavailable - TAVILY_API_KEY not set."
    try:
        response = tavily_client.search(query=query, max_results=3)
        results = []
        for r in response.get("results", []):
            results.append(f"- {r.get('title', 'No title')}: {r.get('content', '')[:200]}")
        return "\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"

# ---- Research Agent (uses Tavily sparingly for current info) ----
research_agent = Agent(
    name="Research Agent",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.5-flash-lite",
        openai_client=client
    ),
    instructions=(
        "You research CURRENT travel information. Use the search tool ONLY ONCE to get essential updates like:\n"
        "- Current travel advisories or restrictions\n"
        "- Recent attraction openings/closures\n"
        "- Current events or festivals during travel dates\n"
        "Do NOT search for general information you already know. Respond concisely in one message."
    ),
    tools=[search_web],
)

# ---- Planner Agent (builds day-by-day itinerary) ----
planner_agent = Agent(
    name="Planner Agent",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.5-flash-lite",
        openai_client=client
    ),
    instructions=(
        "Create a day-by-day travel itinerary. Be concise and respond in one message. "
        "Include key attractions and activities for each day."
    ),
)

# ---- Budget Agent (estimates costs under constraints) ----
budget_agent = Agent(
    name="Budget Agent",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.5-flash-lite",
        openai_client=client
    ),
    instructions=(
        "Estimate travel costs for lodging, food, transport, and activities. Be concise and respond in one message. "
        "Provide a breakdown and total estimate."
    ),
)

# ---- Local Guide Agent (adds local tips & dining) ----
local_guide_agent = Agent(
    name="Local Guide Agent",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.5-flash-lite",
        openai_client=client
    ),
    instructions=(
        "Provide local food recommendations, restaurant suggestions, and cultural tips. Be concise and respond in one message."
    ),
)

# ---- Core orchestrator: Travel Agent ----
travel_agent = Agent(
    name="Travel Agent",
    model=OpenAIChatCompletionsModel(
        model="gemini-2.5-flash-lite",
        openai_client=client
    ),
    instructions=(
        "You orchestrate travel planning by calling these agents IN PARALLEL (all at once, not sequentially):\n"
        "1. planner_agent - for the itinerary\n"
        "2. budget_agent - for cost estimates\n"
        "3. local_guide_agent - for food and local tips\n"
        "4. research_agent - ONLY if user needs current/real-time info (advisories, recent events)\n\n"
        "IMPORTANT: Call agents in a SINGLE turn. Only use research_agent when truly necessary.\n"
        "After receiving their responses, present the combined travel plan with clear sections:\n"
        "## Itinerary\n[planner agent response]\n\n## Budget\n[budget agent response]\n\n## Local Tips\n[local guide response]\n\n## Current Updates\n[research agent response, if used]"
    ),
    tools=[
        planner_agent.as_tool(
            tool_name="planner_agent",
            tool_description="Creates day-by-day itinerary"),
        budget_agent.as_tool(
            tool_name="budget_agent",
            tool_description="Estimates trip costs"),
        local_guide_agent.as_tool(
            tool_name="local_guide_agent",
            tool_description="Provides food and local tips"),
        research_agent.as_tool(
            tool_name="research_agent",
            tool_description="Gets CURRENT travel info (advisories, events). Use sparingly - only when real-time data needed.")
    ]
)

# ---- Sidebar: User Inputs ----
st.sidebar.header("📝 Trip Details")

destination = st.sidebar.text_input(
    "🌍 Destination",
    value="Tokyo",
    help="Enter your travel destination"
)

num_days = st.sidebar.number_input(
    "📅 Number of Days",
    min_value=1,
    max_value=30,
    value=5,
    help="How many days is your trip?"
)

budget = st.sidebar.number_input(
    "💵 Budget (USD)",
    min_value=100,
    max_value=50000,
    value=2000,
    step=100,
    help="Your total trip budget in US dollars"
)

preferences = st.sidebar.text_area(
    "✨ Special Preferences (optional)",
    placeholder="e.g., vegetarian food, museums, adventure activities, family-friendly...",
    help="Any special requirements or interests for your trip"
)

# ---- Agent Info Display ----
with st.sidebar.expander("🤖 Meet the AI Agents"):
    st.markdown("""
    **🧠 Planner Agent**
    Creates your day-by-day itinerary

    **💰 Budget Agent**
    Estimates and tracks all costs

    **🍣 Local Guide Agent**
    Recommends food & local tips

    **🔍 Research Agent**
    Fetches current travel updates (used sparingly)

    **✈️ Travel Agent**
    Orchestrates all agents for the final plan
    """)

# ---- Helper function to parse sections ----
def parse_travel_plan(text: str, destination: str, num_days: int) -> dict:
    """Parse the travel plan text into sections."""
    sections = {
        "destination": destination,
        "duration": f"{num_days} days",
        "itinerary": "",
        "budget": "",
        "tips": "",
        "updates": ""
    }

    # Try to extract sections using regex
    itinerary_match = re.search(r'##\s*Itinerary\s*(.*?)(?=##|$)', text, re.DOTALL | re.IGNORECASE)
    budget_match = re.search(r'##\s*Budget\s*(.*?)(?=##|$)', text, re.DOTALL | re.IGNORECASE)
    tips_match = re.search(r'##\s*(?:Local\s*)?Tips\s*(.*?)(?=##|$)', text, re.DOTALL | re.IGNORECASE)
    updates_match = re.search(r'##\s*(?:Current\s*)?Updates\s*(.*?)(?=##|$)', text, re.DOTALL | re.IGNORECASE)

    if itinerary_match:
        sections["itinerary"] = itinerary_match.group(1).strip()
    if budget_match:
        sections["budget"] = budget_match.group(1).strip()
    if tips_match:
        sections["tips"] = tips_match.group(1).strip()
    if updates_match:
        sections["updates"] = updates_match.group(1).strip()

    # If no sections found, use the full text as itinerary
    if not any([sections["itinerary"], sections["budget"], sections["tips"]]):
        sections["itinerary"] = text

    return sections

# ---- Async Runner Function ----
async def generate_trip_plan(destination: str, num_days: int, budget: int, preferences: str) -> str:
    """Run the multi-agent system to generate a trip plan."""
    prompt = f"""Plan a {num_days}-day trip to {destination} with a budget of ${budget} USD.

{'Special preferences: ' + preferences if preferences else 'No special preferences.'}

Please create a comprehensive travel plan including:
1. A detailed day-by-day itinerary
2. Estimated costs for accommodations, food, transportation, and activities
3. Local food recommendations and cultural tips
"""
    result = await Runner.run(travel_agent, prompt, max_turns=10)
    return result.final_output

# ---- Main App Logic ----
st.divider()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    plan_button = st.button("🚀 Generate Trip Plan", type="primary", use_container_width=True)

if plan_button:
    with st.spinner("🤖 AI agents are working on your trip plan..."):
        # Status placeholders for each agent
        status_container = st.container()
        with status_container:
            st.info("✈️ **Travel Agent** is coordinating the team...")
            planner_status = st.empty()
            budget_status = st.empty()
            guide_status = st.empty()

            planner_status.info("🧠 **Planner Agent** is creating your itinerary...")
            budget_status.info("💰 **Budget Agent** is calculating costs...")
            guide_status.info("🍣 **Local Guide Agent** is finding local gems...")

        try:
            # Run the async function
            raw_result = asyncio.run(generate_trip_plan(destination, num_days, budget, preferences))

            # Parse the result
            result = parse_travel_plan(str(raw_result), destination, num_days)

            # Clear status messages
            status_container.empty()

            st.success("✅ Trip plan generated successfully!")
            st.divider()

            # Display Results
            st.header(f"🗺️ Your {result['duration']} Trip to {result['destination']}")

            # Create tabs for different sections
            tabs = ["📋 Itinerary", "💰 Budget Breakdown", "🍣 Local Tips"]
            if result.get("updates"):
                tabs.append("🔍 Current Updates")

            tab_objects = st.tabs(tabs)

            with tab_objects[0]:
                st.subheader("🧠 Day-by-Day Itinerary")
                st.markdown(result["itinerary"] or "No itinerary available.")

            with tab_objects[1]:
                st.subheader("💰 Cost Estimates")
                st.markdown(result["budget"] or "No budget breakdown available.")

                # Budget comparison
                st.divider()
                st.metric(
                    label="Your Budget",
                    value=f"${budget:,}",
                    delta=None
                )

            with tab_objects[2]:
                st.subheader("🍣 Local Recommendations & Tips")
                st.markdown(result["tips"] or "No local tips available.")

            if result.get("updates") and len(tab_objects) > 3:
                with tab_objects[3]:
                    st.subheader("🔍 Current Travel Updates")
                    st.markdown(result["updates"])

            # Download option
            st.divider()
            full_plan = f"""# {result['destination']} Trip Plan ({result['duration']})

## Itinerary
{result['itinerary']}

## Budget
{result['budget']}

## Local Tips
{result['tips']}
"""
            if result.get("updates"):
                full_plan += f"""
## Current Updates
{result['updates']}
"""
            st.download_button(
                label="📥 Download Trip Plan",
                data=full_plan,
                file_name=f"{destination.lower().replace(' ', '_')}_trip_plan.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Please check your API key and try again.")

# Footer
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray;'>Powered by Multi-Agent AI System</div>",
    unsafe_allow_html=True
)