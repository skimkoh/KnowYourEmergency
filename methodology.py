import streamlit as st
st.set_page_config(layout="wide")

st.header("Methodology", divider="grey")

st.text("There will be two flowcharts - one for the triage assistant, and one for the education assistant.")
st.text("In the flowchart, the components with green background represents an one-off operation to populate guidelines from the data sources.")

st.write("**Triage Assistant Flowchart**")
st.image("flowchart.png", caption="Flowchart for Triage Assistant")

st.write("**Education Assistant Flowchart**")
