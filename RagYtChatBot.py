import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import re 

st.set_page_config(page_title="YouTube Video Chatbot", page_icon="", layout="centered")
st.title("YouTube Video Chatbot")
st.caption("Paste a YouTube link, ask questions about the video's content.")

# ---------- Sidebar: API key ----------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("groq API key", type="password",
                             help="Get one free at https://aistudio.google.com/app/apikey")
    st.markdown("---")
    st.markdown(
        "**Pipeline:** transcript → chunks → HuggingFace embeddings → FAISS → Groq"
    )

# ---------- Session state ----------
for key, default in [
    ("vector_store", None),
    ("video_id", None),
    ("chat_history", []),
    ("main_chain", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def extract_video_id(url_or_id: str) -> str:
    """Pull an 11-char YouTube video id out of a URL, or return the input if it already looks like an id."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url_or_id.strip()):
        return url_or_id.strip()
    return url_or_id.strip()


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vector_store(video_id: str):
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id, languages=["en"])
    transcript = " ".join(snippet.text for snippet in fetched_transcript)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def build_chain(vector_store, api_key: str):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    groq_api_key=api_key
)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=["context", "question"],
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    return parallel_chain | prompt | llm | StrOutputParser()


# ---------- Load video ----------
url_input = st.text_input("YouTube video URL or ID", placeholder="https://www.youtube.com/watch?v=...")
load_clicked = st.button("Load video", type="primary", disabled=not url_input or not api_key)

if not api_key:
    st.info("Enter your Groq API key in the sidebar to get started.")

if load_clicked:
    video_id = extract_video_id(url_input)
    with st.spinner("Fetching transcript and building the index..."):
        try:
            st.session_state.vector_store = build_vector_store(video_id)
            st.session_state.video_id = video_id
            st.session_state.main_chain = build_chain(st.session_state.vector_store, api_key)
            st.session_state.chat_history = []
            st.success("Video loaded. Ask a question below.")
        except TranscriptsDisabled:
            st.error("Captions are disabled for this video.")
        except NoTranscriptFound:
            st.error("No English transcript is available for this video.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

if st.session_state.video_id:
    st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")

# ---------- Chat ----------
if st.session_state.main_chain:
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)

    question = st.chat_input("Ask something about the video...")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.main_chain.invoke(question)
                except Exception as e:
                    answer = f"Error: {e}"
                st.markdown(answer)
        st.session_state.chat_history.append(("assistant", answer))
        