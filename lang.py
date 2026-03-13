from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

key = "I'll paste Key later here"

os.environ["OPENAI_API_KEY"] = key 


# ── 1. Your "knowledge base" (can come from a PDF, website, etc.) ──
company_docs = [
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
    Using the Fitzen App, you can order your food to your location. One plate Biryani costs around Rs.100.
    Cool drinks are complementary with no additional charges.
    """)
]

# ── 2. Split into chunks ──
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(company_docs)

# ── 3. Embed + store ──
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever   = vectorstore.as_retriever(search_kwargs={"k": 2})

# ── 4. RAG prompt ──
rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful customer support agent for FitZen.
Answer ONLY using the context below. If unsure, say "I don't know."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ── 5. Build the RAG chain ──
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

# ── 6. Ask questions! ──
print(rag_chain.invoke("How much does FitZen Premium cost?"))
# → "FitZen Premium costs ₹599/month."

print(rag_chain.invoke("How do I get a refund?"))
# → "Refunds are available within 7 days of billing..."
