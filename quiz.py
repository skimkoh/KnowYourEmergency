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
from populate_chroma import populate_chroma_db

def populate_quiz_items():
    llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o-mini", temperature=0)

    prompt = edu_prompt.format(context=context)

    quiz = llm.invoke(prompt)

    st.session_state.quiz_items = json.loads(quiz.content)


if "vectorstore_loaded" not in st.session_state:
    populate_chroma_db(st.secrets["OPENAI_API_KEY"], persist_dir="chroma_db")
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    st.session_state.vectorstore = Chroma(
        persist_directory="chroma_db",
        collection_name="triage_guide",
        embedding_function=embeddings
    )

    results = st.session_state.vectorstore.get(include=["documents", "metadatas"])
    st.session_state.context = "\n\n".join(results["documents"])
    st.session_state.vectorstore_loaded = True

context = st.session_state.context

edu_prompt = PromptTemplate.from_template("""
You are a medical triage assistant for the public and you are trying to educate the public about when medical situations are considered as a non-emergency or an emergency.
From the following medical guidelines and examples as the context, create 10 short quiz-style questions that test users on whether a situation is an EMERGENCY or NON-EMERGENCY. 
Please also have some new questions (that did not reference the guidelines) based on the context and examples from the guidelines as well.

Each question should have:
- a short symptom scenario (1–2 sentences)
- the correct answer ("EMERGENCY" or "NON-EMERGENCY")
- a one to two sentence explanation. If it is a non emergency, also provide some medical advice. For example if its an ankle injury, there is a RICE method as a form of treatment.

Also ensure:
- Questions are not repeated
- Try to have some questions that are harder
- Avoid making the answer for the questions alternate. (For example, question 1 is emergency, question 2 is non-emergency, question 3 is emergency.)

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

if "quiz_items" not in st.session_state:
    populate_quiz_items()


quiz_items = st.session_state.quiz_items

if "questions" not in st.session_state:
    st.session_state.questions = []   # will load randomly selected dataset
if "index" not in st.session_state:
    st.session_state.index = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "finished" not in st.session_state:
    st.session_state.finished = False


def restart_quiz():
    
    st.session_state.questions = quiz_items[:]  # exact original order
    
    st.session_state.index = 0
    st.session_state.score = 0
    st.session_state.finished = False
    st.session_state.answered = False


def refresh_quiz():
    populate_quiz_items()
    quiz_items = st.session_state.quiz_items

    st.session_state.questions = quiz_items[:]  # exact original order
    
    st.session_state.index = 0
    st.session_state.score = 0
    st.session_state.finished = False
    st.session_state.answered = False



st.title("Emergency or Not Quiz?")

if len(st.session_state.questions) == 0:
    st.info("Click below to start the quiz!")
    if st.button("Start Quiz"):
        restart_quiz()
        st.rerun()

if st.session_state.finished:
    st.subheader("Quiz Completed!")
    st.write(f"Your Score: **{st.session_state.score} / {len(st.session_state.questions)}**")

    if st.button("Restart Same Quiz (This will return the same questions)"):
        restart_quiz()
        st.rerun()

    if st.button("Refresh Quiz (This option will try to generate different questions)"):
        refresh_quiz()
        st.rerun()

if st.session_state.index >= len(st.session_state.questions):
    st.error("Question index out of range. Restarting quiz...")
    restart_quiz()
    st.rerun()

current = st.session_state.questions[st.session_state.index]

if st.session_state.finished == False: 
    st.subheader(f"Question {st.session_state.index + 1} of {len(st.session_state.questions)}")
    st.write("### 🩺 Symptom:")
    st.write(f"> {current['symptom']}")

    choice = st.radio(
        "Is this an emergency?",
        ["EMERGENCY", "NON-EMERGENCY"],
        horizontal=True,
        key=f"choice_{st.session_state.index}"
    )

    if "answered" not in st.session_state:
        st.session_state.answered = False

    if st.button("Submit Answer"):
        if choice == current["answer"]:
            st.success("Correct!")
            st.session_state.score += 1
        else:
            st.error("Incorrect!")

        st.session_state.answered = True
        st.info(f"**Explanation:** {current['explanation']}")

if st.session_state.answered == True:
    if st.session_state.index + 1 < len(st.session_state.questions):
        if st.button("Next Question"):
            st.session_state.answered = False
            st.session_state.index += 1
            st.rerun()
    else:
        if st.session_state.finished != True:
            if st.button("Finish Quiz"):
                st.session_state.finished = True
                st.rerun() 

