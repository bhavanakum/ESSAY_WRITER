import streamlit as st
from essay_functions import (
    AgentState,
    plan_node,
    research_plan_node,
    generation_node,
    reflection_node,
    research_critique_node
)

st.set_page_config(page_title="LLM Essay Generator", layout="centered")
st.title("🧠 Essay Generator")

# User input
user_input = st.text_area("✍️ Enter your essay task:", height=150)

if st.button("Generate Essay") and user_input:
    with st.spinner("Planning..."):
        state = AgentState({"task": user_input, "max_revisions": 3, "revision_number": 1})
        plan = plan_node(state)
        state.update(plan)
        st.subheader("📝 Planning")
        #st.markdown(plan)

    with st.spinner("Researching..."):
        research_plan = research_plan_node(state)
        state.update(research_plan)
        st.subheader("🔍 Researching")
        #st.markdown(research_plan)

    with st.spinner("Generating initial draft..."):
        generation = generation_node(state)
        state.update(generation)
        st.subheader("🖊️ Initial Draft")
        #st.markdown(generation)

    with st.spinner("Reflecting on quality..."):
        reflection = reflection_node(state)
        state.update(reflection)
        st.subheader("🔄 Reflection")
        #st.markdown(reflection)

    with st.spinner("Critiquing with more research..."):
        critique = research_critique_node(state)
        state.update(critique)
        st.subheader("🔍 Critique")
        #st.markdown(critique)

    with st.spinner("Finalizing improved essay..."):
        final_generation = generation_node(state)
        state.update(final_generation)
        st.subheader("🎯 Final Essay")
        #st.markdown(final_generation)

    # Final Essay
    st.subheader("📄 Final Essay")
    st.markdown(state["draft"])
