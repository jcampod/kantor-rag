import streamlit as st
from pinecone import Pinecone
from groq import Groq
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# Page config
st.set_page_config(
    page_title="J.R. Kantor Research System",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Image URLs from GitHub
IMAGE_URL = "https://raw.githubusercontent.com/jcampod/kantor-rag/main/kantor.png"

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
    
    /* Search input styling - WHITE background */
    .stTextInput > div > div > input {{
        border: 2px solid #ddd;
        border-radius: 25px;
        padding: 0.8rem 1.2rem;
        font-family: 'Georgia', serif;
        font-size: 1rem;
        width: 100%;
        background-color: #ffffff !important;
    }}
    
    .stTextInput > div > div {{
        background-color: transparent !important;
    }}
    
    .stTextInput > div {{
        background-color: transparent !important;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: #b8232f;
        box-shadow: 0 0 0 2px rgba(184, 35, 47, 0.15);
        background-color: #ffffff !important;
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
    
    /* Download button - WHITE text */
    .stDownloadButton > button {{
        background-color: #b8232f !important;
        color: white !important;
        border-radius: 4px;
    }}
    
    .stDownloadButton > button span {{
        color: white !important;
    }}
    
    .stDownloadButton > button p {{
        color: white !important;
    }}
    
    .stDownloadButton > button div {{
        color: white !important;
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


def diversify_results(matches, max_per_source=2):
    """
    Limit results to max N chunks per source document.
    Keeps the highest-scoring chunks from each source.
    Results are already sorted by score (highest first) from Pinecone.
    """
    source_counts = defaultdict(int)
    diversified = []
    
    for match in matches:
        source = match.metadata.get('filename', 'Unknown')
        
        if source_counts[source] < max_per_source:
            diversified.append(match)
            source_counts[source] += 1
    
    return diversified


# Document catalog
DOCUMENT_CATALOG = {
    "Books": [
        "1959.Interbehavioral Psychology - J. R. Kantor",
        "A Survey of the Science of Psyc - J. R. Kantor",
        "An Outline of Social Psychology - J. R. Kantor",
        "Interbehavioral philosophy - J. R. Kantor",
        "Principles of psychology Vol 1 - J. R. Kantor",
        "Principles of psychology Vol 2 - J. R. Kantor",
        "Psychology and Logic Vol 1 - J. R. Kantor",
        "Psychology and Logic Vol 2 - J. R. Kantor",
        "The Aim and Progress of Psychology and Other Sciences",
        "The Scientific Evolution of Psychology Vol II",
        "The Scientific evolution of Psy - J. R. Kantor",
        "the science of psychology an interbehavioral survey",
    ],
    "Articles": [
        "1917 Intelligence and mental tests",
        "1917. Discussion. Statistics of - J. R. Kantor",
        "1917_The functional nature of the philosophical categories",
        "1918_Conscious behavior and the abnormal",
        "1918_The Ethics of Internationalism and the Individual",
        "1919. Instrumental transformism - J. R. Kantor",
        "1919_Human Personality and its Pathology",
        "1919_Psychology as a science of critical evaluation",
        "1920_A functional interpretation of human instincts",
        "1920_Intelligence and Mental Tests",
        "1920_Suggestions toward a scientific interpretation of perception",
        "1921_An Objective Interpretation of Meanings",
        "1921_An attempt toward a naturalistic description of emotions I",
        "1921_An attempt toward a naturalistic description of emotions II",
        "1921_Association as a fundamental process of objective psychology",
        "1921_How do we acquire our basic reactions",
        "1922 Can the psychophysical experiment reconcile introspectionists and relativists",
        "1922_An analysis of psychological language data",
        "1922_An essay toward an institutional conception of social psychology",
        "1922_How is a science of social psychology possible",
        "1922_The Nervous System, Psychological Fact or Fiction",
        "1922_The Psychology of Reflex Action",
        "1922_The integrative character of habits",
        "1923_An objective analysis of volitional behavior",
        "1923_The Institutional Foundation of a Scientific Social Psychology",
        "1923_The problem of instinct and its relation to social psychology",
        "1923_The psychology of feeling or affective reactions",
        "1925_Anthropology, race, psychiatry, and culture",
        "1925_The Significance of the Gestalt Conception in Psychology",
        "1928 Can psychology contribute to the study of linguistics",
        "1933_A survey of the science of psychology",
        "1933_In defense of stimulus-response psychology",
        "1935_James Mark Baldwin Columbia, S. C., 1861--Paris, France, 1934",
        "1935_The evolution of mind",
        "1936 Concerning Physical Analogies in Psychology",
        "1938_Character and personality. Their nature and interrelations",
        "1938_The nature of psychology as a natural science",
        "1938_The operational principle in the physical and psychological sciences",
        "1939_The current situation in social psychology",
        "1941_Current trends in psychological theory",
        "1942_Toward a scientific analysis of motivation",
        "1945_Problems and paradoxes of physiological psychology",
        "1956_Interbehavioral psychology and scientific analysis of data and operations",
        "1957_Events and constructs in the science of psychology",
        "1959_Evolution and the science of psychology",
        "1960_History of science as scientific method",
        "1962_Psychology Scientific status-seeker",
        "1963_Behaviorism whose image",
        "1964_History of psychology What benefits",
        "1968_Behaviorism in the history of psychology",
        "1969_Scientific psychology and specious philosophy",
        "1970_An analysis of the experimental analysis of behavior (TEAB)",
        "1970_Innate intelligence Another genetic avatar",
        "1970_Newton's influence on the development of psychology",
        "1971_Revivalism in psychology",
        "1973_Private data, raw feels, inner experience, and all that",
        "1973_System structure and scientific psychology",
        "1973_The hereditarian manifesto (politics in psychology)",
        "1974_Eppur si muove",
        "1974_Lest we forget",
        "1974_The role of chemistry in the domain of psychology",
        "1975 La lingüística psicológica",
        "1975_Education in psychological perspective",
        "1975_In dispraise of indiscrimination",
        "1975_On reviewing psychological classics",
        "1976_Behaviorism, behavior analysis, and the career of psychology",
        "1976_Cultural institutions and psychological institutions",
        "1976_What meaning means in linguistics",
        "1978 Experimentation the ACME of science",
        "1978_Cognition as events and as psychic constructions",
        "1978_Man and machines in psychology Cybernetics and artificial intelligence",
        "1979_Psychology Science or nonscience",
        "1980_Manifesto of interbehavioral psychology",
        "1980_Theological psychology vs. scientific psychology",
        "1981 Axioms and their role in psychology",
        "1981 reflections upon speech and language",
        "1981_Interbehavioral psychology and the logic of science",
        "1981_Surrogation A process in psychological evolution",
        "1982 Objectivity and subjectivity in science and psychology",
        "1982_Psychological retardation and interbehavioral maladjustments",
        "1984_Scientific unity and spiritistic disunity",
        "1984_The relation of scientists to events in physics and in psychology",
    ],
    "Reviews": [
        "1925_[Review of] Lucien Levy-Bruhl.Primitive Mentality",
        "1936_[Review of] Gestalt Psychology A Survey of Facts and Principles",
        "1937_[Review of] Holmes, R. W. The idealism of Giovanni Gentile",
        "1937_[Review of] Langer, S. K. An introduction to symbolic logic",
        "1938_[Review of] Burloud, A. Principe d'une psychologie des tendances",
        "1938_[Review of] Köhler, W. The place of value in the world of facts",
        "1939_[Review of] Gray, L. H. Foundations of Language",
        "1939_[Review of] Moore, T. V. Cognitive psychology",
        "1939_[Review of] What Man Has Made of Man A Study of the Consequences of Platonism and Positivism in Psychology",
        "1940_[Review of] Eddington, S. A. The philosophy of physical science",
        "1941_[Review of] Wood, L. The analysis of knowledge",
        "1943_[Review of] Thorndike, E. L. Man and his works",
        "1943_[Review of] Tolman, E. C. Drives toward war",
        "1944_[Review of] Hunt, J. M. (Ed.). Personality and the behavior disorders A handbook based on experimental and clinical research",
        "1954_[Review of] Bentley, A. F. Inquiry into inquiries Essays in social theory",
        "1966_[Review of] Alexander, F.; Eisenstein, S.; & Grotjahn, M. (Eds.). Psychoanalytic pioneers",
        "1966_[Review of] Platt, J. R. (Ed.). New views of the nature of man",
        "1968_[Review of] Watson, R. I. The great psychologists From Aristotle to Freud.",
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

# Search input with button
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

# Trigger search on button click or Enter key
if (search_clicked or query) and query:
    with st.spinner("Searching..."):
        try:
            query_embedding = model.encode(query).tolist()
            
            # Build filter
            pinecone_filter = None
            if doc_type != "All Types":
                pinecone_filter = {"doc_type": {"$eq": doc_type}}
                if title_filter and not title_filter.startswith("All ") and title_filter != "Select type first":
                    pinecone_filter = {
                        "$and": [
                            {"doc_type": {"$eq": doc_type}},
                            {"filename": {"$eq": title_filter}}
                        ]
                    }
            
            # Query more results initially to allow for diversification
            results = index.query(
                namespace="",
                vector=query_embedding,
                top_k=25,
                include_metadata=True,
                filter=pinecone_filter
            )
            
            # Apply source diversification: max 2 chunks per document
            diversified_matches = diversify_results(results.matches, max_per_source=2)
            
            # Limit to top 10 after diversification
            diversified_matches = diversified_matches[:10]
            
            context = ""
            sources = []
            source_references = ""
            
            for i, match in enumerate(diversified_matches, 1):
                text = match.metadata.get("text", "")
                filename = match.metadata.get("filename", "Unknown")
                page = match.metadata.get("page", "?")
                doc_type_result = match.metadata.get("doc_type", "")
                
                context += f"\n[Source {i}: {filename}, p.{page}]\n{text}\n"
                source_references += f"- Source {i}: {filename}, page {page}\n"
                
                sources.append({
    "num": i,
    "file": filename,
    "title": filename,
    "type": doc_type_result,
    "page": page,
    "year": match.metadata.get("year", 0),
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
                    year_badge = f"({s['year']}) " if s.get('year') else ""
with st.expander(f"Source {s['num']}: {type_badge}{year_badge}{s['title']} — p.{s['page']} ({s['score']:.0%})"):
                        st.markdown(f'<div class="source-text">{s["text"]}</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown('<p class="footer-caption">20 Books • 91 Articles • 21 Reviews • 1915–1984</p>', unsafe_allow_html=True)
st.markdown('<p class="footer-caption"><a href="https://interbehavioral.com/contact/" target="_blank" style="color: #b8232f;">Provide feedback</a></p>', unsafe_allow_html=True)
