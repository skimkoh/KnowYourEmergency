import streamlit as st
st.set_page_config(layout="wide")
st.header("Sample Pictures for Medical Triage Assistant", divider="grey")

st.text("As shown in the features section in the About Us section, users have the ability to upload an image for the assistant to evalute the seriousness of the symptom.")
st.text("Here are some images which can be used.")
st.text("To use them, right click and download the image.")

st.image("sample_pictures/cut.jpeg")
st.image("sample_pictures/bruise.jpg")
st.image("sample_pictures/smile.jpg")
