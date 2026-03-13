import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# ── API Key ──
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── Page Config ──
st.set_page_config(
    page_title="FitZen Support",
    page_icon="🧘",
    layout="centered"
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,400;0,600;1,400&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #fdf8f2;
    }

    /* Header */
    .fitzen-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .fitzen-header h1 {
        font-family: 'Fraunces', serif;
        font-size: 2.6rem;
        color: #2d2d2d;
        margin-bottom: 0.2rem;
    }
    .fitzen-header p {
        color: #7a6f65;
        font-size: 1rem;
    }

    /* Chat messages */
    .chat-bubble {
        padding: 0.85rem 1.1rem;
        border-radius: 14px;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        line-height: 1.6;
        max-width: 85%;
    }
    .user-bubble {
        background-color: #e8f4ef;
        color: #1a3329;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    .bot-bubble {
        background-color: #fff8f0;
        color: #2d2d2d;
        border: 1px solid #eddfc8;
        border-bottom-left-radius: 4px;
    }
    .bubble-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
        color: #a8998a;
    }

    /* API Key input */
    .stTextInput > div > div > input[type="password"] {
        background-color: #fff8f0;
        border: 1.5px solid #ddd0bb;
        border-radius: 8px;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Chat input */
    .stChatInput > div {
        border: 1.5px solid #ddd0bb !important;
        border-radius: 12px !important;
        background-color: #fffcf8 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #fff8f0;
        border-right: 1px solid #eddfc8;
    }
    [data-testid="stSidebar"] h2 {
        font-family: 'Fraunces', serif;
        color: #2d2d2d;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #6dab8a !important;
    }

    /* Divider */
    hr {
        border-color: #eddfc8;
    }
</style>
""", unsafe_allow_html=True)


# ── Knowledge Base ──
COMPANY_DOCS = [
    Document(page_content="""
    FitZen is a fitness app founded in 2022. Our mission is to make
    personalized fitness accessible to everyone. We offer yoga, HIIT,
    and meditation programs. Monthly subscription is ₹299.
    """),
    Document(page_content="""
    FitZen's cancellation policy: Users can cancel anytime from Settings →
    Subscription → Cancel. Refunds are available within 7 days of billing.
    Contact support@fitzen.in for help.
    """),
    Document(page_content="""
    FitZen Premium includes offline downloads, custom meal plans, and
    live sessions with certified trainers. Premium costs ₹599/month.
    """),
    Document(page_content="""
    Using the Fitzen App, you can order your food to your location. One plate
    Biryani costs around Rs.100. Cool drinks are complementary with no additional charges.
    """)
]

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful customer support agent for FitZen.
Answer ONLY using the context below. If unsure, say "I currently do not have the information regarding this. However, we are trying our best to assist you. You contact the customer support number '123456788' if you need specific details regarding."

Context:
{context}

Question: {question}
""")


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


@st.cache_resource(show_spinner=False)
def build_rag_chain():
    """Build and cache the RAG chain."""

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(COMPANY_DOCS)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | ChatOpenAI(model="gpt-4o-mini")
        | StrOutputParser()
    )
    return chain


# ── Sidebar ──
with st.sidebar:
    st.markdown("## 💡 Try asking")
    sample_questions = [
        "How much does FitZen Premium cost?",
        "How do I get a refund?",
        "What programs does FitZen offer?",
        "Can I order food on FitZen?",
        "How do I cancel my subscription?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Header ──
st.markdown("""
<div class="fitzen-header">
    <h1>🧘 FitZen Support</h1>
    <p>Ask me anything about FitZen — plans, refunds, features & more.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Session State ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ── Render Chat History ──
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-end; margin-bottom:0.8rem;">
            <div>
                <div class="bubble-label" style="text-align:right;">You</div>
                <div class="chat-bubble user-bubble">{msg["content"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="margin-bottom:0.8rem;">
            <div class="bubble-label">FitZen Support 🤖</div>
            <div class="chat-bubble bot-bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Small Talk Handler ──
SMALL_TALK = {
    ("hello", "hi", "hey", "howdy", "hiya"):
        "Hello! 👋 Welcome to FitZen Support. How can I help you today?",
    ("how are you", "how are you doing", "how do you do"):
        "I'm doing great, thanks for asking! 😊 How can I assist you with FitZen today?",
    ("good morning",):
        "Good morning! ☀️ How can I help you with FitZen today?",
    ("good evening",):
        "Good evening! 🌙 How can I help you with FitZen today?",
    ("good afternoon",):
        "Good afternoon! How can I help you with FitZen today?",
    ("bye", "goodbye", "see you", "see ya", "take care"):
        "Goodbye! 👋 Have a great day and stay fit with FitZen! 🧘",
    ("thanks", "thank you", "thank you so much", "thanks a lot"):
        "You're welcome! 😊 Feel free to ask if you have any other questions.",
}

def get_small_talk_response(text: str):
    lowered = text.strip().lower().rstrip("!.,?")
    for triggers, reply in SMALL_TALK.items():
        if lowered in triggers:
            return reply
    return None


# ── Handle a question (from input OR sidebar button) ──
def answer_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})

    # Render user bubble immediately
    st.markdown(f"""
    <div style="display:flex; justify-content:flex-end; margin-bottom:0.8rem;">
        <div>
            <div class="bubble-label" style="text-align:right;">You</div>
            <div class="chat-bubble user-bubble">{question}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    small_talk = get_small_talk_response(question)
    if small_talk:
        response = small_talk
    else:
        with st.spinner("Thinking..."):
            try:
                chain = build_rag_chain()
                response = chain.invoke(question)
            except Exception as e:
                response = f"❌ Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown(f"""
    <div style="margin-bottom:0.8rem;">
        <div class="bubble-label">FitZen Support 🤖</div>
        <div class="chat-bubble bot-bubble">{response}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar Button Trigger ──
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    answer_question(q)

# ── Chat Input ──
user_input = st.chat_input("Ask a question about FitZen…")
if user_input:
    answer_question(user_input)