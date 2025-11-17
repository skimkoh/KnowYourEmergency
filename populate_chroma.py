import os
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

# --------- function to fetch text from URL ---------
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


def populate_chroma_db(openapi_api_key: str, persist_dir="chroma_db"):
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("ChromaDB already exists — skipping population.")
        return


    print("Populating ChromaDB for first time...") 
    llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

    url_1 = "https://www.channelnewsasia.com/cna-insider/scdf-emergency-ambulance-services-995-calls-life-death-hospital-4180921"  
    url_2 = "https://www.scdf.gov.sg/home/about-scdf/emergency-medical-services#:~:text=The%20SCDF%20responded%20to%20256%2C837,bleeding%2C%20major%20traumas%20and%20stroke"

    article_1 = fetch_url_text(url_1)
    article_2 = fetch_url_text(url_2)

    if not article_1 or not article_2:
        st.error("Could not fetch one or both articles. Please check URLs.")
        st.stop()


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

    vectorstore = Chroma(
        persist_directory="chroma_db",
        collection_name="triage_guide",
        embedding_function=embeddings
    )
    vectorstore.add_documents(chunks)
    vectorstore.persist()