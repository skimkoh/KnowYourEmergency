import streamlit as st
import requests
import os
import tempfile
import base64
import httpx
import openai
import json

from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.llms import OpenAI
from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from PIL import Image # Required for working with image data
from sentence_transformers import SentenceTransformer, util  # For text similarity
from langchain_core.messages import HumanMessage

# --------- Function to fetch text from URL ---------
def fetch_url_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        # Get visible text
        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return text
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return ""

url_1 = "https://www.channelnewsasia.com/cna-insider/scdf-emergency-ambulance-services-995-calls-life-death-hospital-4180921"  
url_2 = "https://www.scdf.gov.sg/home/about-scdf/emergency-medical-services#:~:text=The%20SCDF%20responded%20to%20256%2C837,bleeding%2C%20major%20traumas%20and%20stroke"

article_1 = fetch_url_text(url_1)
article_2 = fetch_url_text(url_2)

if not article_1 or not article_2:
    st.error("Could not fetch one or both articles. Please check URLs.")
    st.stop()

llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])


summary_prompt = PromptTemplate.from_template("""
Extract key facts, entities, and context from the following articles.
Return bullet points covering symptoms of a hospital emergency and symptoms of a non emergency, and what should be done for either an emergency or a non-emergency (example: hotline, where to go)

Article:
{article}
""")


summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

summary_1 = summary_chain.run(article=article_1)
summary_2 = summary_chain.run(article=article_2)

docs = [
    Document(page_content=summary_1, metadata={"source": "article_1"}),
    Document(page_content=summary_2, metadata={"source": "article_2"})
]

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

if not os.path.exists("chroma_db") or not os.listdir("chroma_db"):
    vectorstore = Chroma(
        persist_directory="chroma_db",
        collection_name="triage_guide",
        embedding_function=embeddings
    )
    vectorstore.add_documents(chunks)
    vectorstore.persist()
  
else:
    vectorstore = Chroma(
        persist_directory="chroma_db",
        collection_name="triage_guide",
        embedding_function=embeddings
        
    )
 

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

 # --- confirmation prompt ---

# prompt displayed to the user, based on the infomation gathered from the user (input / image) - show a summary of what the user is suffering from
confirmation_prompt = PromptTemplate.from_template("""
You are a medical triage assistant for the public.
You will receive a user's description or a image (or both) of their symptoms or situation.

Act like a doctor and confirm and summarize the medical symptoms that they are suffering from.
If you are unable to detect any medical symptoms based on their input, then let the user know that you are unable to triage anything and submit again with more clearer and detailed medical symptoms.

If you are confident that based on the detailed symptoms that the user is not suffering from any medical symptoms, state clearly to the user that they are healthy.

Avoid giving medical advices or asking more information from users if you are able to summarize the symptoms.
Avoid ending with something like this: "If you have any additional symptoms or concerns, please consider seeking medical attention", since you just need to state the summarized symptoms

Context:
{situation}

""")

confirmation_chain = LLMChain(llm=llm, prompt=confirmation_prompt)

 # --- triage prompt ---

triage_prompt = PromptTemplate.from_template("""
You are a medical triage assistant for the public.
You will receive a user's description of their symptoms or situation.

Use the context below (guidelines about emergencies vs non-emergencies)
to determine:
1. Whether it’s an **EMERGENCY** or **NON-EMERGENCY**
2. What the user should do next (e.g., call 995, call 1777, visit GP, self-care or use the SMS services) based on the context of both articles.
3. If it is a non emergency and based on the seriousness of the user's condition, also advice the user to call 1777 for an non emergency ambulance based on your discretion 

If uncertain, err on the side of safety.

Context:
{summaries}

Question:
{question}

Answer in this JSON format:
{{
  "category": "EMERGENCY" | "NON-EMERGENCY",
  "advice": "string explaining next steps clearly"
}}

""")

triage_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": triage_prompt, "document_variable_name": "summaries"}
)



st.title("🚑 :red[Emergency Triage Assistant]")

with st.form("symptom_form"):
    st.write("**Unsure whether your medical symptoms are considered as emergencies or non-emergencies? Use this form to help you make better decisions!**")
    st.write("**To start using this triage assistant, you can input either your symptoms (or medical situation) AND/or upload a photo of your symptoms / situation.**")

    user_input = st.text_area(label="Describe your symptoms or situation:", placeholder="Type your symptoms or situation..." )
    uploaded_image = st.file_uploader("Upload an image of your symptoms or situation", type=["png","jpg","jpeg"], accept_multiple_files=False)


    submitted = st.form_submit_button("Submit")


if submitted:
    if not user_input and not uploaded_image:
        st.error("Please either input some symptoms or upload a file of your symptoms before submitting.")
    else:
        with st.spinner("Analyzing symptoms..."):

             # --- Extract text from image if uploaded ---
             
            image_text = "" 
            if uploaded_image:

                prompt = "Describe any visible medical symptoms from the picture. If there are no medical symptoms, return a generic response that says that there are no symptoms detected"
                openai.api_key = st.secrets["OPENAI_API_KEY"]
                
                image_bytes = uploaded_image.read()
                base64_data = base64.b64encode(image_bytes).decode("utf-8")
                mime_type = uploaded_image.type 
                data_url = f"data:{mime_type};base64,{base64_data}"

                # send image and prompt in a single user message
                response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ]   
                )

                image_text = response.choices[0].message.content


            # --- Handle different input combinations ---
            if user_input and image_text:
                # Compare text similarity using embeddings
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embeddings = model.encode([user_input, image_text], convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

                # Threshold for contradiction (tune as needed)
                if similarity < 0.6:
                    
                    st.subheader("Summarized symptoms from your input", divider="red")

                    image_text_summary = confirmation_chain.run(situation=image_text)
                    st.write("**Medical symptoms from image:**")
                    st.write(image_text_summary)
                    st.write('For reference, this is your uploaded image:')
                    image = Image.open(uploaded_image)

                    st.image(image, caption="Uploaded Image", use_container_width=True)

                    st.divider()
                    st.write("**Medical symptoms from free-text field:**")
                    user_input_summary = confirmation_chain.run(situation=user_input)
                    st.write(user_input_summary)

                    st.divider()
                    st.subheader("Triage Assistant Advice:", divider="red")
                    st.text("Traige assisant is unable to give proper advice as the medical symptoms from the free-text field and the uploaded image seem to contradict each other. Please clarify or re-upload another image.")
                
                else:

                    # the image and the free-text input are relevant to each other, use both to make a decision
                    st.subheader("Summarized symptoms from your inputs", divider="red")

                    image_text_summary = confirmation_chain.run(situation=image_text)
                    st.write("**Medical symptoms from image:**")
                    st.write(image_text_summary)
                    st.write('For reference, this is your uploaded image:')
                    image = Image.open(uploaded_image)

                    st.image(image, caption="Uploaded Image", use_container_width=True)

                    st.divider()
                    st.write("**Medical symptoms from free-text field:**")
                    user_input_summary = confirmation_chain.run(situation=user_input)
                    st.write(user_input_summary)


                    st.divider()
                    st.subheader("Triage Assistant Advice:", divider="blue")
                    combined_input = user_input + " " + image_text
                    result = triage_chain.invoke({"question": combined_input})
                    parsed_results = json.loads(result['answer'])
                    st.write(f"This is a **{parsed_results['category']}**.")
                    st.write(f"{parsed_results['advice']}")

            else:
                # Only text or only image
                final_input = user_input if user_input else image_text
                
                st.subheader("Summarized symptoms from your inputs", divider="red")
                if user_input:
                    st.write("**Medical symptoms from free-text field:**")
                    user_input_summary = confirmation_chain.run(situation=user_input)
                    st.write(user_input_summary)


                    st.divider()
                    st.subheader("Triage Assistant Advice:", divider="blue")
                    combined_input = user_input + " " + image_text
                    result = triage_chain.invoke({"question": combined_input})
                    parsed_results = json.loads(result['answer'])
                    st.write(f"This is a **{parsed_results['category']}**.")
                    st.write(f"{parsed_results['advice']}")

                if image_text:
                    image_text_summary = confirmation_chain.run(situation=image_text)
                    st.write("**Medical symptoms from image:**")
                    st.write(image_text_summary)
                    st.write('For reference, this is your uploaded image:')
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                    st.divider()
                    st.subheader("Triage Assistant Advice:", divider="blue")
                    combined_input = user_input + " " + image_text
                    result = triage_chain.invoke({"question": combined_input})
                    parsed_results = json.loads(result['answer'])
                    st.write(f"This is a **{parsed_results['category']}**.")
                    st.write(f"{parsed_results['advice']}")

                   