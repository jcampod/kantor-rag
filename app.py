import streamlit as st
from pinecone import Pinecone
from groq import Groq
from sentence_transformers import SentenceTransformer
import base64

# Page config
st.set_page_config(
    page_title="J.R. Kantor Research System",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS - BLACK color scheme
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 950px;
    }
    
    /* Custom header with image */
    .custom-header {
        background: linear-gradient(135deg, #1a1a1a 0%, #333333 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        display: flex;
        align-items: center;
        gap: 2rem;
    }
    
    .header-image {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
        border: 3px solid rgba(255,255,255,0.3);
    }
    
    .header-text h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .header-text p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* Search input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #333;
        box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling - BLACK */
    .stButton > button {
        background: linear-gradient(135deg, #333333 0%, #1a1a1a 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Answer section */
    .answer-box {
        background: #f8fafc;
        border-left: 4px solid #333;
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Source text preview */
    .source-text {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 1rem;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #333;
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* Source header */
    .source-header {
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    
    /* Example questions */
    .example-btn {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .example-btn:hover {
        background: #e2e8f0;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] > div {
        background: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# Initialize clients
@st.cache_resource
def init_clients():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("kantor-rag")
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return index, groq_client, model

index, groq_client, model = init_clients()

# Load and encode image
@st.cache_data
def get_image_base64():
    try:
        with open("kantor.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

img_base64 = get_image_base64()

# Custom header with image
if img_base64:
    st.markdown(f"""
    <div class="custom-header">
        <img src="data:image/png;base64,{img_base64}" class="header-image">
        <div class="header-text">
            <h1>J.R. Kantor Research System</h1>
            <p>Search through Kantor's complete works on interbehavioral psychology</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="custom-header">
        <div class="header-text">
            <h1>J.R. Kantor Research System</h1>
            <p>Search through Kantor's complete works on interbehavioral psychology</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This research tool provides access to J.R. Kantor's 
    complete academic bibliography on interbehavioral psychology.
    """)
    
    st.markdown("---")
    
    st.markdown("### Collection")
    st.markdown("""
    - **130** documents indexed  
    - Books, articles, and reviews  
    - **1915 - 1984**
    """)
    
    st.markdown("---")
    
    st.markdown("### How to use")
    st.markdown("""
    1. Type your question below
    2. Click **Search**
    3. Review the AI-generated answer
    4. Explore source documents
    5. Download results if needed
    """)

# Example questions
st.markdown("**Try an example:**")
col1, col2 = st.columns(2)

example_clicked = None
with col1:
    if st.button("What is interbehavioral psychology?", key="ex1"):
        example_clicked = "What is interbehavioral psychology?"
    if st.button("How does Kantor define stimulus?", key="ex2"):
        example_clicked = "How does Kantor define stimulus?"

with col2:
    if st.button("What is the interbehavioral field?", key="ex3"):
        example_clicked = "What is the interbehavioral field?"
    if st.button("Kantor's view on language", key="ex4"):
        example_clicked = "What is Kantor's view on language?"

st.markdown("---")

# Search input
query = st.text_input(
    "Enter your research question:",
    value=example_clicked if example_clicked else "",
    placeholder="e.g., What is the interbehavioral field?",
    label_visibility="visible"
)

# Search button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    search_clicked = st.button("Search", use_container_width=True)

if search_clicked or (query and example_clicked):
    if query:
        with st.spinner("Searching Kantor's works..."):
            try:
                # Generate query embedding
                query_embedding = model.encode(query).tolist()
                
                # Search Pinecone - 10 results
                results = index.query(
                    namespace="default",
                    vector=query_embedding,
                    top_k=10,
                    include_metadata=True
                )
                
                # Build context and sources list
                context = ""
                sources = []
                source_references = ""
                
                for i, match in enumerate(results.matches, 1):
                    text = match.metadata.get("text", "")
                    filename = match.metadata.get("filename", "Unknown")
                    title = match.metadata.get("title", filename)
                    page = match.metadata.get("page", "?")
                    doc_type = match.metadata.get("type", "")
                    
                    # Build context with source markers
                    context += f"\n[Source {i}: {title}, p.{page}]\n{text}\n"
                    
                    # Build source reference list
                    source_references += f"- Source {i}: {title}, page {page}\n"
                    
                    sources.append({
                        "num": i,
                        "file": filename,
                        "title": title,
                        "type": doc_type,
                        "page": page,
                        "score": match.score,
                        "text": text
                    })
                
                # Generate response
                if context.strip():
                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": f"""You are a scholar specializing in J.R. Kantor's interbehavioral psychology. 
Answer based ONLY on the provided context. 
When citing, use the format [Source X] to reference specific sources.

Available sources:
{source_references}

Always cite which source(s) your information comes from using [Source X] notation.
If the context doesn't contain relevant information, say so."""
                            },
                            {
                                "role": "user",
                                "content": f"Context:\n{context}\n\nQuestion: {query}"
                            }
                        ]
                    )
                    
                    answer = response.choices[0].message.content
                    
                    # Display answer
                    st.markdown("### Answer")
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                    
                    # Download/Copy buttons
                    st.markdown("---")
                    col_a, col_b, col_c = st.columns([1, 1, 2])
                    
                    # Prepare download content
                    download_text = f"""QUERY: {query}

ANSWER:
{answer}

SOURCES:
"""
                    for s in sources:
                        download_text += f"\n{'='*60}\nSource {s['num']}: {s['title']} — Page {s['page']}\nType: {s['type']}\nRelevance: {s['score']:.1%}\n{'='*60}\n{s['text']}\n"
                    
                    with col_a:
                        st.download_button(
                            label="Download Results",
                            data=download_text,
                            file_name=f"kantor_search_{query[:30].replace(' ', '_')}.txt",
                            mime="text/plain"
                        )
                    
                    with col_b:
                        if st.button("Copy Answer"):
                            st.code(answer, language=None)
                            st.caption("Select and copy the text above")
                    
                else:
                    st.warning("No relevant documents found for this query.")
                
                # Display sources with text preview
                if sources:
                    st.markdown("### Sources")
                    st.caption("Click on each source to view the retrieved text")
                    
                    for s in sources:
                        type_badge = f"[{s['type']}] " if s['type'] else ""
                        with st.expander(f"Source {s['num']}: {type_badge}{s['title']} — Page {s['page']} ({s['score']:.1%})"):
                            st.markdown(f'<div class="source-text">{s["text"]}</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
