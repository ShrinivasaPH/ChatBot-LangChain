import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
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
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #fdf8f2; }
    .fitzen-header { text-align: center; padding: 2rem 0 1rem; }
    .fitzen-header h1 { font-family: 'Fraunces', serif; font-size: 2.6rem; color: #2d2d2d; margin-bottom: 0.2rem; }
    .fitzen-header p { color: #7a6f65; font-size: 1rem; }
    .fitzen-header .badge {
        display: inline-block; background: #fff0e0; border: 1px solid #e8cfa0;
        color: #b07d3a; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em;
        text-transform: uppercase; padding: 0.2rem 0.7rem; border-radius: 20px; margin-top: 0.5rem;
    }
    .chat-bubble { padding: 0.85rem 1.1rem; border-radius: 14px; margin-bottom: 0.5rem; font-size: 0.95rem; line-height: 1.6; max-width: 85%; }
    .user-bubble { background-color: #e8f4ef; color: #1a3329; margin-left: auto; border-bottom-right-radius: 4px; }
    .bot-bubble { background-color: #fff8f0; color: #2d2d2d; border: 1px solid #eddfc8; border-bottom-left-radius: 4px; }
    .bubble-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 0.3rem; color: #a8998a; }
    .stChatInput > div { border: 1.5px solid #ddd0bb !important; border-radius: 12px !important; background-color: #fffcf8 !important; }
    [data-testid="stSidebar"] { background-color: #fff8f0; border-right: 1px solid #eddfc8; }
    .stSpinner > div { border-top-color: #6dab8a !important; }
    hr { border-color: #eddfc8; }
</style>
""", unsafe_allow_html=True)


# ── Knowledge Base ──
COMPANY_DOCS = [
    Document(page_content="""
    FitZen is a fitness app founded in 2022. Our mission is to make
    personalized fitness accessible to everyone. We offer yoga, HIIT,
    and meditation programs. Monthly subscription is Rs.299.
    """),
    Document(page_content="""
    FitZen's cancellation policy: Users can cancel anytime from Settings ->
    Subscription -> Cancel. Refunds are available within 7 days of billing.
    Contact support@fitzen.in for help.
    """),
    Document(page_content="""
    FitZen Premium includes offline downloads, custom meal plans, and
    live sessions with certified trainers. Premium costs Rs.599/month.
    """),
    Document(page_content="""
    Using the Fitzen App, you can order your food to your location. One plate
    Biryani costs around Rs.100. Cool drinks are complementary with no additional charges.
    """)
]

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a warm, friendly customer support agent for FitZen.

Guidelines:
- Answer using the context provided below whenever possible.
- If the user's message is a short follow-up reaction (like "Really?", "Are you sure?", "Wow!",
  "Tell me more", "That's great!", "Interesting!", "Makes sense", "Oh okay"), use the conversation
  history to give a natural, confirming, or elaborating response - exactly like a human would.
- If the question is completely outside the context and you genuinely cannot answer, respond humbly.
  For example: "That's a great question! I'm afraid I don't have that information right now.
  For further help, feel free to reach out to support@fitzen.in"
  Never bluntly say "I don't know."
- Keep responses warm, concise, and human.

Conversation so far:
{history}

Context from FitZen knowledge base:
{context}

User: {question}
Assistant:""")


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def format_history(messages: list) -> str:
    recent = messages[-6:] if len(messages) > 6 else messages
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "FitZen Support"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines) if lines else "No prior conversation."


@st.cache_resource(show_spinner=False)
def build_retriever():
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(COMPANY_DOCS)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})


def run_rag(question: str, history: list) -> str:
    retriever = build_retriever()
    docs = retriever.invoke(question)
    context = format_docs(docs)
    hist_text = format_history(history)
    prompt_value = RAG_PROMPT.invoke({
        "history": hist_text,
        "context": context,
        "question": question
    })
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(prompt_value)
    return StrOutputParser().invoke(response)


# ── Small Talk Handler ──
SMALL_TALK = {
    ("hello", "hi", "hey", "howdy", "hiya"):
        "Hello! Welcome to FitZen Support. How can I help you today?",
    ("how are you", "how are you doing", "how do you do"):
        "I'm doing great, thanks for asking! How can I assist you with FitZen today?",
    ("good morning",): "Good morning! How can I help you with FitZen today?",
    ("good evening",): "Good evening! How can I help you with FitZen today?",
    ("good afternoon",): "Good afternoon! How can I help you with FitZen today?",
    ("bye", "goodbye", "see you", "see ya", "take care"):
        "Goodbye! Have a great day and stay fit with FitZen!",
    ("thanks", "thank you", "thank you so much", "thanks a lot"):
        "You're welcome! Feel free to ask if you have any other questions.",
}

def get_small_talk_response(text: str):
    lowered = text.strip().lower().rstrip("!.,?")
    for triggers, reply in SMALL_TALK.items():
        if lowered in triggers:
            return reply
    return None


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:

    # ── About Card ──
    st.markdown("""
    <div style="background:#fff0e0; border:1px solid #e8cfa0; border-radius:10px; padding:0.9rem 1rem 1rem; margin-bottom:0.8rem;">
        <div style="font-size:0.68rem; font-weight:700; letter-spacing:0.09em; text-transform:uppercase; color:#b07d3a; margin-bottom:0.35rem;">Academic Project</div>
        <div style="font-size:1rem; font-weight:700; color:#2d2d2d; font-family:'Georgia', serif;">Shrinivasa PH</div>
        <div style="font-size:0.8rem; color:#7a6f65; margin-top:0.4rem; line-height:1.5;">
            A demo chatbot built to showcase Retrieval-Augmented Generation (RAG) using LangChain.<br><br>
            <span style="font-style:italic; color:#b07d3a;">FitZen is a fictional company created solely for this demonstration.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tech Stack ──
    with st.expander("⚙️ Tech Stack & Concepts", expanded=False):
        st.markdown("""
**🔗 LangChain**
A framework for building LLM-powered apps. Connects prompts, models, retrievers, and parsers into clean, composable pipelines.

---
**📚 RAG — Retrieval-Augmented Generation**
Instead of relying solely on the LLM's training data, RAG first *retrieves* relevant documents and passes them as context — grounding answers in your actual knowledge base.

---
**🔢 Embeddings**
Text is converted into numerical vectors (lists of floats) capturing *semantic meaning*. Similar sentences cluster together in this vector space, enabling meaning-based search — not just keyword matching.

---
**🗄️ Vector Store — FAISS**
A database optimised for storing and searching embedding vectors. A user's question is embedded and compared against stored vectors to instantly find the most relevant document chunks.

---
**⛓️ LCEL — LangChain Expression Language**
A declarative `|` pipe syntax for chaining components:
```
retriever | prompt | llm | parser
```
Each step feeds output into the next — readable and composable.

---
**🤖 LLM — GPT-4o Mini**
OpenAI's efficient model that generates the final response, given retrieved context and conversation history.
        """)

    st.markdown("---")

    # ── Sample Questions ──
    st.markdown("**💡 Try asking**")
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


# ══════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════

# ── Header ──
st.markdown("""
<div class="fitzen-header">
    <h1>🧘 FitZen Support</h1>
    <p>Ask me anything about FitZen (a fictional company).</p>
    <p><em>Plans, Refunds, Features & More.</em></p>
    <span class="badge">Academic Demo &nbsp;·&nbsp; Shrinivasa PH</span>
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


# ── Handle a question ──
def answer_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})

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
                history_so_far = st.session_state.messages[:-1]
                response = run_rag(question, history_so_far)
            except Exception as e:
                response = f"Error: {str(e)}"

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