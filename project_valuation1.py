import streamlit as st
from typing import TypedDict, List
from dotenv import load_dotenv
import numpy_financial as npf

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# -----------------------------
# Load environment (optional)
# -----------------------------
load_dotenv()

# -----------------------------
# State Definition
# -----------------------------
class FinanceState(TypedDict):
    initial_investment: float
    fcff: List[float]
    discount_rate: float

    npv: float
    irr: float
    payback: float
    roce: float
    explanation: str


# -----------------------------
# Finance Metrics Node
# -----------------------------
def finance_metrics_node(state: FinanceState):
    fcff = state["fcff"]
    r = state["discount_rate"]
    initial = state["initial_investment"]

    # NPV
    npv = -initial + sum(
        fcff[t] / ((1 + r) ** (t + 1)) for t in range(len(fcff))
    )

    # IRR
    irr = npf.irr([-initial] + fcff)

    # Payback Period
    cumulative = 0
    payback = None
    for i, cash in enumerate(fcff):
        cumulative += cash
        if cumulative >= initial:
            prev = cumulative - cash
            payback = i + (initial - prev) / cash
            break

    # ROCE (FCFF proxy)
    avg_profit = sum(fcff) / len(fcff)
    roce = avg_profit / initial

    return {
        **state,
        "npv": round(npv, 2),
        "irr": round(irr * 100, 2),
        "payback": round(payback, 2),
        "roce": round(roce * 100, 2),
    }


# -----------------------------
# Explanation Node (Groq)
# -----------------------------
def explanation_node(state: FinanceState, llm):
    prompt = f"""
    You are a senior corporate finance analyst.

    Explain the following project evaluation results:

    - NPV: {state['npv']}
    - IRR: {state['irr']}%
    - Payback Period: {state['payback']} years
    - ROCE: {state['roce']}%

    Keep it concise, professional, and decision-oriented.
    Do NOT invent numbers.
    """

    explanation = llm.invoke(prompt).content
    return {**state, "explanation": explanation}


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Project Evaluation (FCFF)", layout="wide")

st.title("📊 Project Evaluation using FCFF")
st.caption("LangGraph + Groq | NPV • IRR • Payback • ROCE")

# Sidebar – Groq API Key
st.sidebar.header("🔑 Groq Configuration")
groq_api_key = st.sidebar.text_input(
    "Enter your GROQ API Key",
    type="password"
)

# Inputs
col1, col2 = st.columns(2)

with col1:
    initial_investment = st.number_input(
        "Initial Investment",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    discount_rate = st.number_input(
        "Discount Rate (WACC)",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.01
    )

with col2:
    fcff_input = st.text_area(
        "FCFF (comma-separated)",
        value="3000, 3200, 3500, 3800, 4000"
    )

# Convert FCFF
try:
    fcff = [float(x.strip()) for x in fcff_input.split(",")]
except:
    st.error("Invalid FCFF input")
    st.stop()

# Run Button
if st.button("🚀 Evaluate Project"):

    if not groq_api_key:
        st.error("Please enter your GROQ API Key")
        st.stop()

    # Initialize LLM
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    # Build Graph
    graph = StateGraph(FinanceState)

    graph.add_node("finance_metrics", finance_metrics_node)
    graph.add_node(
        "explanation",
        lambda s: explanation_node(s, llm)
    )

    graph.set_entry_point("finance_metrics")
    graph.add_edge("finance_metrics", "explanation")
    graph.add_edge("explanation", END)

    app = graph.compile()

    # Invoke
    result = app.invoke({
        "initial_investment": initial_investment,
        "fcff": fcff,
        "discount_rate": discount_rate
    })

    # Results
    st.subheader("📈 Financial Results")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("NPV", f"{result['npv']}")
    r2.metric("IRR (%)", f"{result['irr']}")
    r3.metric("Payback (Years)", f"{result['payback']}")
    r4.metric("ROCE (%)", f"{result['roce']}")

    st.subheader("🧠 Analyst Explanation")
    st.write(result["explanation"])