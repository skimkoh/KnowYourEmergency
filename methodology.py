import streamlit as st
st.set_page_config(layout="wide")

st.header("Methodology", divider="grey")

st.text("There will be two flowcharts - one for the triage assistant, and one for the education assistant.")
st.info("In the flowchart, the components with green background represents an one-off operation to populate structured guidelines from the scrapped data sources, if the structured data cannot be found")

st.write("**Triage Assistant Flowchart**")
st.image("flowcharts/flowchart_triage.png", caption="Flowchart for Triage Assistant")

st.write("**Education Assistant Flowchart**")
st.image("flowcharts/flowchart_quiz.png", caption="Flowchart for Education Assistant", width=600)
