import streamlit as st
from pinecone import Pinecone
from groq import Groq
from sentence_transformers import SentenceTransformer

# Page config
st.set_page_config(
    page_title="J.R. Kantor Research System",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .custom-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .custom-header p {
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
        border-color: #2c5282;
        box-shadow: 0 0 0 3px rgba(44, 82, 130, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2c5282 0%, #1a365d 100%);
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
        box-shadow: 0 4px 12px rgba(26, 54, 93, 0.3);
    }
    
    /* Answer section */
    .answer-box {
        background: #f8fafc;
        border-left: 4px solid #2c5282;
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    /* Source expanders */
    .streamlit-expanderHeader {
        background: #f1f5f9;
        border-radius: 6px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #f8fafc;
    }
    
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

# Custom header
st.markdown("""
<div class="custom-header">
    <h1>J.R. Kantor Research System</h1>
    <p>Search through Kantor's complete works on interbehavioral psychology</p>
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
    """)

# Search input
query = st.text_input(
    "Enter your research question:",
    placeholder="e.g., What is the interbehavioral field?",
    label_visibility="visible"
)

# Search button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    search_clicked = st.button("Search", use_container_width=True)

if search_clicked or query:
    if query:
        with st.spinner("Searching Kantor's works..."):
            try:
                # Generate query embedding
                query_embedding = model.encode(query).tolist()
                
                # Search Pinecone
                results = index.query(
                    namespace="default",
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True
                )
                
                # Build context
                context = ""
                sources = []
                
                for match in results.matches:
                    text = match.metadata.get("text", "")
                    context += f"\n{text}\n"
                    sources.append({
                        "file": match.metadata.get("filename", "Unknown"),
                        "page": match.metadata.get("page", "?"),
                        "score": match.score
                    })
                
                # Generate response
                if context.strip():
                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a scholar specializing in J.R. Kantor's interbehavioral psychology. Answer based ONLY on the provided context. Always cite the source documents. If the context doesn't contain relevant information, say so."
                            },
                            {
                                "role": "user",
                                "content": f"Context:\n{context}\n\nQuestion: {query}"
                            }
                        ]
                    )
                    
                    answer = response.choices[0].message.content
                    
                    st.markdown("### Answer")
                    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                else:
                    st.warning("No relevant documents found for this query.")
                
                # Display sources
                if sources:
                    st.markdown("### Sources")
                    for i, s in enumerate(sources, 1):
                        with st.expander(f"{i}. {s['file']} — Page {s['page']}"):
                            st.caption(f"Relevance: {s['score']:.1%}")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
