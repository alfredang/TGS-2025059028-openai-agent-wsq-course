# Activity: Build a Multi-Agent AI Trip Planner (Agent-as-Tools Pattern)

## Overview

In this activity, you will design a **Multi-Agent AI System** using the **Agent-as-Tools** pattern with the `.as_tool()` method. Each specialized agent has **Tavily Search tools** for accurate, real-time information.

**App Name:** 🧳 AI Trip Planner (Multi-Agent System)

**Sample Query:** "Plan a 5-day trip to Tokyo under $2,000 with food recommendations"

---

## Learning Objectives

By the end of this activity, you will:
1. Understand the **Agent-as-Tools** pattern using `.as_tool()` method
2. Learn how to create specialized agents with `handoff_description`
3. Implement **Tavily Search tools** for real-time information
4. Build an orchestrator that intelligently delegates to specialized agents
5. Use Pydantic models for structured output

---

## Architecture: Agent-as-Tools with Tavily Search

The Travel Agent orchestrates specialized agents as tools:
- **planner_agent.as_tool()** → 🧠 Planner Agent (search_attractions)
- **budget_agent.as_tool()** → 💰 Budget Agent (search_accommodation_prices, search_transport_costs)
- **local_guide_agent.as_tool()** → 🍣 Local Guide Agent (search_food_restaurants, search_local_tips)

---

## AI Agents with Tavily Search Tools

### 🧠 Planner Agent
- **Tool:** `planner_agent`
- **Search Tool:** `search_attractions` - Finds tourist attractions, landmarks, activities
- **Output:** Day-by-day itinerary with real attractions

### 💰 Budget Agent
- **Tool:** `budget_agent`
- **Search Tools:**
  - `search_accommodation_prices` - Finds hotel/hostel costs
  - `search_transport_costs` - Finds transportation prices
- **Output:** Budget breakdown with real prices

### 🍣 Local Guide Agent
- **Tool:** `local_guide_agent`
- **Search Tools:**
  - `search_food_restaurants` - Finds restaurants and local food
  - `search_local_tips` - Finds cultural tips and travel advice
- **Output:** Food recommendations and local tips

### ✈️ Travel Agent (Orchestrator)
- **Role:** Coordinates all agents using `.as_tool()`
- **Output:** Structured `TravelOutput` model

---

## How to Run

1. Ensure you have your environment variables set:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

2. Run the application:
   ```bash
   streamlit run app/trip_planner.py
   ```

3. Open your browser to `http://localhost:8501`

---

## Key Concepts

### 1. Tavily Search Integration
Each agent has specialized search tools that query Tavily for real-time information.

### 2. Agent-as-Tools Pattern
Convert agents to tools using `.as_tool()`:
```python
planner_agent.as_tool(
    tool_name="planner_agent",
    tool_description="Plan itineraries with real attractions"
)
```

### 3. Structured Output
Use Pydantic models for typed responses:
```python
class TravelOutput(BaseModel):
    destination: str
    duration: str
    summary: str
    cost: str
    tips: str
```

---

## Benefits

| Benefit | Description |
|---------|-------------|
| **Real-Time Data** | Tavily search provides current, accurate information |
| **Specialized Tools** | Each agent has domain-specific search capabilities |
| **Clean Orchestration** | `.as_tool()` pattern for elegant agent coordination |
| **Structured Output** | Pydantic models ensure typed, predictable responses |

---

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
- [Tavily API Documentation](https://tavily.com/docs)
