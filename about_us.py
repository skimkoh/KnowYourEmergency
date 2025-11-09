import streamlit as st
st.set_page_config(layout="wide")
st.header("About Us - Medical Triage Assistant", divider="grey")

st.subheader("Project Scope")

st.text("In 2024, statistics report released by the Singapore Civil Defence Force (SCDF) showed that it received 245,279 emergency medical calls in 2024, an average of 672 calls daily. Of these, nearly five per cent or 10,728 of the calls were non-emergencies, a slight uptick compared with 2023.")
st.text("Responding to non-emergency medical calls diverts critical resources – ambulances, paramedics and medical supplies – away from those who need urgent help. And when someone has suffered a cardiac arrest, for example, every second counts.")
st.text("This project aims to develop a prototype AI assistant that helps the public determine whether their symptoms indicate an emergency or non-emergency condition.")
st.divider()

st.subheader("Objectives")

st.markdown(
    """
    - **Symptom Classification**
        - Analyze user-entered medical symptoms using natural language processing
        - Users can choose to input a short description of their medical symptoms, and/or upload a URL link of their symptoms
        - The assistant will take in both (or one) input and classify the situation as an EMERGENCY or NON-EMERGENCY

    - **Guided Advice**
        - Recommend next steps (e.g., call 995, call 1777, visit a GP, or self-care)
        - Provide concise and empathetic advice written in plain language
    
    - **Educational Engagement**
        - 'Education Assistant' where users can learn to differentiate emergency vs non-emergency symptoms through a MCQ quiz
    """
)

st.text("The goal is not to replace professional diagnosis, but to educate users and assist users in the responsible use of emergency services — ensuring that emergency hotlines and ambulances are reserved for true emergencies, while others receive proper self-care or non-emergency advice.")

st.divider()

st.subheader("Data Sources")

st.text("There are no available datasets of medical diagonsis since medical cases are confidental. Therefore, the data sources used for this project are mostly scrapped via SCDF's official website on Emergency Medical Services, as well as news platform such as CNA which covers articles on non-emergencies and emergencies.")

data_source_table = {
    "Data Source": [
        "SCDF (Singapore Civil Defence Force)",
        "Channel NewsAsia (CNA) article",
        "Internal ChromaDB Collection",
    ],
    "Description": ["Official emergency medical guidelines", "Real-world context of ambulance usage and public awareness", "Summarized emergency/non-emergency scenarios (generated from LLM processing of articles)"],
    "Purpose": ["Distinguish between emergency and non-emergency cases", "Provide example cases and best practices", "Used for vector retrieval and reasoning"],
    "URLs (if any)": [1247, 892, 654],
}

st.table(data_source_table, border="horizontal")

st.divider()

st.subheader("Key Features")

st.write("**Triage Assistant")
st.text("User input their medical symptoms with either/both a free-text area and a picture uploader.")
st.text("The assistant will first summarize the symptoms to the users, then give advice as to whether their symptoms are considered as non-emergency / emergency and give next steps.")

st.





