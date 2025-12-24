import streamlit as st
from pinecone import Pinecone
from groq import Groq
from sentence_transformers import SentenceTransformer

# Page config
st.set_page_config(
    page_title="J.R. Kantor Research System",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Image URLs from GitHub
IMAGE_URL = "https://raw.githubusercontent.com/jcampod/kantor-rag/main/kantor.png"
SEARCH_ICON_URL = "https://raw.githubusercontent.com/jcampod/kantor-rag/main/search.png"

# Custom CSS - Professional academic design with RED theme
st.markdown(f"""
<style>
    /* Hide ALL Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden !important;}}
    header {{visibility: hidden;}}
    .stDeployButton {{display: none !important;}}
    div[data-testid="stDecoration"] {{display: none !important;}}
    div[data-testid="stStatusWidget"] {{display: none !important;}}
    .viewerBadge_container__r5tak {{display: none !important;}}
    .styles_viewerBadge__CvC9N {{display: none !important;}}
    div[data-testid="stToolbar"] {{display: none;}}
    
    /* Main container */
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 900px;
    }}
    
    /* Academic Typography */
    h1, h2, h3 {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #111111;
        font-weight: 600;
    }}
    
    p, div, label, span {{
        font-family: 'Georgia', serif;
        color: #333333;
        line-height: 1.6;
    }}
    
    /* Custom header with RED theme */
    .custom-header {{
        background: linear-gradient(135deg, #b8232f 0%, #8b1a23 100%);
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    
    .header-text h1 {{
        margin: 0;
        font-size: 1.6rem;
        font-weight: 600;
        color: white;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    
    .header-text p {{
        margin: 0.4rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
        color: white;
        font-family: 'Georgia', serif;
    }}
    
    .header-image {{
        width: 90px;
        height: 90px;
        border-radius: 50%;
        object-fit: cover;
        object-position: center top;
        border: 2px solid rgba(255,255,255,0.3);
        margin-left: 1.5rem;
    }}
    
    /* Search box container */
    .search-wrapper {{
        position: relative;
        margin-bottom: 1rem;
    }}
    
    /* Search input styling */
    .stTextInput > div > div > input {{
        border: 2px solid #ddd;
        border-radius: 25px;
        padding: 0.8rem 3.5rem 0.8rem 1.2rem;
        font-family: 'Georgia', serif;
        font-size: 1rem;
        width: 100%;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: #b8232f;
        box-shadow: 0 0 0 2px rgba(184, 35, 47, 0.15);
    }}
    
    /* Hide the search button column visually but keep functionality */
    .search-btn-col button {{
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }}
    
    /* Button styling - RED theme */
    .stButton > button {{
        background-color: #b8232f !important;
        color: white !important;
        border: none !important;
        border-radius: 25px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .stButton > button:hover {{
        background-color: #8b1a23 !important;
        color: white !important;
    }}
    
    .stButton > button p {{
        color: white !important;
    }}
    
    /* Search icon inside button */
    .search-icon-btn {{
        width: 24px;
        height: 24px;
    }}
    
    /* Answer section with RED accent */
    .answer-box {{
        background: #fafafa;
        border-left: 3px solid #b8232f;
        padding: 1.25rem;
        border-radius: 0 4px 4px 0;
        margin: 1rem 0;
        font-family: 'Georgia', serif;
        line-height: 1.7;
    }}
    
    /* Source text preview */
    .source-text {{
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 4px;
        padding: 1rem;
        font-size: 0.9rem;
        font-family: 'Georgia', serif;
        line-height: 1.6;
        color: #333;
        max-height: 280px;
        overflow-y: auto;
    }}
    
    /* Select boxes */
    .stSelectbox > div > div {{
        font-family: 'Georgia', serif;
        border-radius: 4px;
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 0.95rem;
    }}
    
    /* Footer caption */
    .footer-caption {{
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 1.5rem;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    
    /* Filter label */
    .filter-label {{
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 0.3rem;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    
    /* Download button */
    .stDownloadButton > button {{
        background-color: #b8232f !important;
        color: white !important;
        border-radius: 4px;
    }}
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

# Document catalog
DOCUMENT_CATALOG = {
    "Books": [
        "A Survey of the Science of Psyc - J. R. Kantor",
        "An Objective Psychology of Gram - J. R. Kantor",
        "An Outline of Social Psychology - J. R. Kantor",
        "Interbehavioral Psychology - J. R. Kantor",
        "Interbehavioral philosophy - J. R. Kantor",
        "Linguistica Psicologica - J. R. Kantor",
        "Principles of psychology Vol 1 - J. R. Kantor",
        "Principles of psychology Vol 2 - J. R. Kantor",
        "Psychological linguistics - J. R. Kantor",
        "Psychology and Logic Vol 1 - J. R. Kantor",
        "Psychology and Logic Vol 2 - J. R. Kantor",
        "Sketch of J. R. Kantor's Psycho - J. R. Kantor",
        "The Aim and Progress of Psychology and Other Sciences",
        "The Scientific Evolution of Psy - J. R. Kantor",
        "The Scientific Evolution of Psychology Vol II",
        "The Scientific evolution of Psy - J. R. Kantor",
        "The logic of modern science",
        "Un esbozo de Psicologia Social - J. R. Kantor",
        "kantor psicologia interconductu - J. R. Kantor",
        "the science of psychology an interbehavioral survey",
    ],
    "Articles": [
        "A functional interpretation of human instincts",
        "A survey of the science of psychology",
        "An Objective Interpretation of Meanings",
        "An analysis of psychological language data",
        "An analysis of the experimental analysis of behavior (TEAB)",
        "An attempt toward a naturalistic description of emotions I",
        "An attempt toward a naturalistic description of emotions II",
        "An essay toward an institutional conception of social psychology",
        "An objective analysis of volitional behavior",
        "Anthropology, race, psychiatry, and culture",
        "Association as a fundamental process of objective psychology",
        "Axioms and their role in psychology",
        "Behaviorism in the history of psychology",
        "Behaviorism whose image",
        "Behaviorism, behavior analysis, and the career of psychology",
        "Can psychology contribute to the study of linguistics",
        "Can the psychophysical experiment reconcile introspectionists and relativists",
        "Character and personality. Their nature and interrelations",
        "Cognition as events and as psychic constructions",
        "Concerning Physical Analogies in Psychology",
        "Conscious behavior and the abnormal",
        "Cultural institutions and psychological institutions",
        "Current trends in psychological theory",
        "Die interbehavioristische Logik und die zeitgenössische Physik",
        "Education in psychological perspective",
        "Eppur si muove",
        "Events and constructs in the science of psychology",
        "Evolution and the science of psychology",
        "Experimentation the ACME of science",
        "Feelings and emotions as scientific events",
        "History of psychology What benefits",
        "History of science as scientific method",
        "How do we acquire our basic reactions",
        "How is a science of social psychology possible",
        "Human Personality and its Pathology",
        "In defense of stimulus-response psychology",
        "In dispraise of indiscrimination",
        "Innate intelligence Another genetic avatar",
        "Intelligence and mental tests",
        "Interbehavioral psychology and scientific analysis of data and operations",
        "Interbehavioral psychology and the logic of science",
        "James Mark Baldwin Columbia, S. C., 1861--Paris, France, 1934",
        "La lingüística psicológica",
        "Lest we forget",
        "Man and machines in psychology Cybernetics and artificial intelligence",
        "Manifesto of interbehavioral psychology",
        "Newton's influence on the development of psychology",
        "Objectivity and subjectivity in science and psychology",
        "On reviewing psychological classics",
        "Private data, raw feels, inner experience, and all that",
        "Problems and paradoxes of physiological psychology",
        "Psychological retardation and interbehavioral maladjustments",
        "Psychology Science or nonscience",
        "Psychology Scientific status-seeker",
        "Psychology as a science of critical evaluation",
        "Revivalism in psychology",
        "Scientific psychology and specious philosophy",
        "Scientific unity and spiritistic disunity",
        "Suggestions toward a scientific interpretation of perception",
        "Surrogation A process in psychological evolution",
        "System structure and scientific psychology",
        "The Ethics of Internationalism and the Individual",
        "The Institutional Foundation of a Scientific Social Psychology",
        "The Nervous System, Psychological Fact or Fiction",
        "The Psychology of Reflex Action",
        "The Significance of the Gestalt Conception in Psychology",
        "The current situation in social psychology",
        "The evolution of mind",
        "The functional nature of the philosophical categories",
        "The hereditarian manifesto (politics in psychology)",
        "The integrative character of habits",
        "The nature of psychology as a natural science",
        "The operational principle in the physical and psychological sciences",
        "The problem of instinct and its relation to social psychology",
        "The psychology of feeling or affective reactions",
        "The psychology of the ethically rational",
        "The relation of scientists to events in physics and in psychology",
        "The role of chemistry in the domain of psychology",
        "Theological psychology vs. scientific psychology",
        "Toward a scientific analysis of motivation",
        "What meaning means in linguistics",
        "reflections upon speech and language",
    ],
    "Reviews": [
        "[Review of] Alexander, F.; Eisenstein, S.; & Grotjahn, M. (Eds.). Psychoanalytic pioneers",
        "[Review of] Bentley, A. F. Inquiry into inquiries Essays in social theory",
        "[Review of] Burloud, A. Principe d'une psychologie des tendances",
        "[Review of] Dantzig, T. Aspects of science",
        "[Review of] Eddington, S. A. The philosophy of physical science",
        "[Review of] Gestalt Psychology A Survey of Facts and Principles",
        "[Review of] Gray, L. H. Foundations of Language",
        "[Review of] Hartshorne, C. The philosophy and psychology of sensation",
        "[Review of] Holmes, R. W. The idealism of Giovanni Gentile",
        "[Review of] Hunt, J. M. (Ed.). Personality and the behavior disorders",
        "[Review of] Köhler, W. The place of value in the world of facts",
        "[Review of] Langer, S. K. An introduction to symbolic logic",
        "[Review of] Lucien Levy-Bruhl.Primitive Mentality",
        "[Review of] Moore, T. V. Cognitive psychology",
        "[Review of] Platt, J. R. (Ed.). New views of the nature of man",
        "[Review of] Smith, T. V. Beyond conscience",
        "[Review of] Thorndike, E. L. Man and his works",
        "[Review of] Tolman, E. C. Drives toward war",
        "[Review of] Watson, R. I. The great psychologists From Aristotle to Freud.",
        "[Review of] What Man Has Made of Man",
        "[Review of] Wood, L. The analysis of knowledge",
    ],
}

# Custom header with image on RIGHT - RED theme
st.markdown(f"""
<div class="custom-header">
    <div class="header-text">
        <h1>J.R. Kantor Research System</h1>
        <p>Search through Kantor's complete works on interbehavioral psychology</p>
    </div>
    <img src="{IMAGE_URL}" class="header-image">
</div>
""", unsafe_allow_html=True)

# Search input with custom icon button
search_col, btn_col = st.columns([8, 1])

with search_col:
    query = st.text_input(
        "Search",
        placeholder="Enter your research question...",
        label_visibility="collapsed"
    )

with btn_col:
    search_clicked = st.button("⌕", help="Search", use_container_width=True)

# Filters BELOW search
st.markdown('<p class="filter-label">Filters (optional)</p>', unsafe_allow_html=True)
filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    doc_type = st.selectbox(
        "Document Type",
        ["All Types", "Books", "Articles", "Reviews"],
        index=0,
        label_visibility="collapsed"
    )

with filter_col2:
    if doc_type != "All Types":
        titles = ["All " + doc_type] + sorted(DOCUMENT_CATALOG.get(doc_type, []))
        title_filter = st.selectbox(
            "Specific Document",
            titles,
            index=0,
            label_visibility="collapsed"
        )
    else:
        title_filter = st.selectbox(
            "Specific Document",
            ["Select type first"],
            index=0,
            disabled=True,
            label_visibility="collapsed"
        )

st.markdown("---")

# Also trigger search on Enter key
if (search_clicked or query) and query:
    with st.spinner("Searching..."):
        try:
            query_embedding = model.encode(query).tolist()
            
            # Build filter
            pinecone_filter = None
            if doc_type != "All Types":
                pinecone_filter = {"type": {"$eq": doc_type}}
                if title_filter and not title_filter.startswith("All ") and title_filter != "Select type first":
                    pinecone_filter = {
                        "$and": [
                            {"type": {"$eq": doc_type}},
                            {"title": {"$eq": title_filter}}
                        ]
                    }
            
            results = index.query(
                namespace="default",
                vector=query_embedding,
                top_k=10,
                include_metadata=True,
                filter=pinecone_filter
            )
            
            context = ""
            sources = []
            source_references = ""
            
            for i, match in enumerate(results.matches, 1):
                text = match.metadata.get("text", "")
                filename = match.metadata.get("filename", "Unknown")
                title = match.metadata.get("title", filename)
                page = match.metadata.get("page", "?")
                doc_type_result = match.metadata.get("type", "")
                
                context += f"\n[Source {i}: {title}, p.{page}]\n{text}\n"
                source_references += f"- Source {i}: {title}, page {page}\n"
                
                sources.append({
                    "num": i,
                    "file": filename,
                    "title": title,
                    "type": doc_type_result,
                    "page": page,
                    "score": match.score,
                    "text": text
                })
            
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
                
                st.markdown("### Answer")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                
                # Download button
                download_text = f"""QUERY: {query}
FILTERS: Type={doc_type}, Document={title_filter if title_filter else 'All'}

ANSWER:
{answer}

SOURCES:
"""
                for s in sources:
                    download_text += f"\n{'='*60}\nSource {s['num']}: [{s['type']}] {s['title']} — Page {s['page']}\nRelevance: {s['score']:.1%}\n{'='*60}\n{s['text']}\n"
                
                st.download_button(
                    label="📥 Download Results",
                    data=download_text,
                    file_name=f"kantor_search.txt",
                    mime="text/plain"
                )
                
            else:
                st.warning("No relevant documents found. Try adjusting your filters or query.")
            
            # Sources
            if sources:
                st.markdown("### Sources")
                for s in sources:
                    type_badge = f"[{s['type']}] " if s['type'] else ""
                    with st.expander(f"Source {s['num']}: {type_badge}{s['title']} — p.{s['page']} ({s['score']:.0%})"):
                        st.markdown(f'<div class="source-text">{s["text"]}</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown('<p class="footer-caption">20 Books • 91 Articles • 21 Reviews • 1915–1984</p>', unsafe_allow_html=True)
