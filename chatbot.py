import streamlit as st
import os
import json
import random
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.documents import Document

st.text('Education Assistant')

embeddings = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


vectorstore = Chroma(
    persist_directory="chroma_db",
    collection_name="triage_guide",
    embedding_function=embeddings
)

results = vectorstore.get(include=["documents", "metadatas"])

# Combine all text into one context
context = "\n\n".join(results["documents"])


edu_prompt = PromptTemplate.from_template("""
You are a medical triage assistant for the public and you are trying to educate the public about when medical situations are considered as a non-emergency or an emergency.
From the following medical guidelines and examples as the context, create 10 short quiz-style questions that test users on whether a situation is an EMERGENCY or NON-EMERGENCY. 
Please also have some new questions (that did not reference the guidelines) based on the context and examples from the guidelines as well.

Each question should have:
- a short symptom scenario (1–2 sentences)
- the correct answer ("EMERGENCY" or "NON-EMERGENCY")
- a one to two sentence explanation. If it is a non emergency, also provide some medical advice.

Return ONLY valid JSON, without markdown or extra text like the example below:
[
  {{
    "symptom": "...", 
    "answer": "EMERGENCY",
    "explanation": "..."
   }},
  ...
]

Guidelines:
{context}
""")

llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o-mini", temperature=0)

prompt = edu_prompt.format(context=context)

quiz = llm.invoke(prompt)

try: 
    quiz_items = json.loads(quiz.content)
except:
    st.error("Unable to parse the generated quiz items. Please refresh and try again")
    st.text(quiz)
    st.stop()


# --- Store quiz state ---
if "current_q" not in st.session_state:
    st.session_state.current_q = random.choice(quiz_items)
    st.session_state.answered = False

q = st.session_state.current_q
st.subheader("🩹 Symptom Scenario:")
st.info(q["symptom"])

# --- Ask the user ---
if not st.session_state.answered:
    guess = st.radio("What do you think this is?", ["EMERGENCY", "NON-EMERGENCY"])
    if st.button("Check Answer"):
        st.session_state.answered = True
        st.session_state.user_guess = guess
       
else:
    if st.session_state.user_guess == q["answer"]:
        st.success(f"✅ Correct! This is an **{q['answer']}** case.")
    else:
        st.error(f"❌ Not quite. This is actually an **{q['answer']}** case.")
    st.write(f"**Explanation:** {q['explanation']}")

    # if st.button("Next Question ➡️"):
    #     st.session_state.current_q = random.choice(quiz_items)
    #     st.session_state.answered = False
    #     st.experimental_rerun()
