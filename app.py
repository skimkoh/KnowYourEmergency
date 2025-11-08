import streamlit as st
from utility import check_password

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()


pages = {
    "Features": [
        st.Page("user_input.py", title="Triage Assistant"),
        st.Page("chatbot.py", title="Education Assistant"),

    ],

     "Documentation": [
        st.Page("about_us.py", title="About Us"),
        st.Page("methodology.py", title="Methodology"),
    ]
}

pg = st.navigation(pages)
pg.run()