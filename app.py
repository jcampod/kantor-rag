import streamlit as st
from pinecone import Pinecone
from groq import Groq

# Page config
st.set_page_config(
    page_title="J.R. Kantor Research System",
    page_icon="📚",
    layout="wide"
)

# Initialize clients
@st.cache_resource
def init_clients():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("kantor-rag")
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return index, groq_client

index, groq_client = init_clients()

# Main UI
st.title("📚 J.R. Kantor Research System")
st.markdown("Search through Kantor's complete works on interbehavioral psychology")

# Debug: Show index stats
with st.sidebar:
    st.markdown("### About")
    st.markdown("This system searches through J.R. Kantor's complete academic works on interbehavioral psychology.")
    st.markdown("### Collection")
    st.markdown("- 130 documents indexed")
    st.markdown("- Books, articles, and reviews")
    st.markdown("- Years: 1915-1984")
    
    # Debug info
    with st.expander("Debug Info"):
        try:
            stats = index.describe_index_stats()
            st.write("Index stats:", stats)
        except Exception as e:
            st.write("Stats error:", str(e))

# Search input
query = st.text_input(
    "Ask a question about Kantor's work:",
    placeholder="e.g., What is interbehavioral psychology?"
)

if st.button("🔍 Search", type="primary") or query:
    if query:
        with st.spinner("Searching..."):
            try:
                # Try inference-enabled search
                results = index.search(
                    namespace="",
                    query={
                        "inputs": {"text": query},
                        "top_k": 5
                    },
                    fields=["text", "filename", "page"]
                )
                
                st.write("DEBUG - Raw results:", results)
                
                # Build context
                context = ""
                sources = []
                
                if "result" in results and "hits" in results["result"]:
                    for match in results["result"]["hits"]:
                        fields = match.get("fields", {})
                        text = fields.get("text", "")
                        context += f"\n{text}\n"
                        sources.append({
                            "file": fields.get("filename", "Unknown"),
                            "page": fields.get("page", "?"),
                            "score": match.get("_score", 0)
                        })
                
                if not sources:
                    st.warning("No documents retrieved. Check index configuration.")
                
                # Generate response only if we have context
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
                    
                    # Display answer
                    st.markdown("### Answer")
                    st.write(answer)
                else:
                    st.markdown("### Answer")
                    st.write("No relevant documents found in the index.")
                
                # Display sources
                st.markdown("### Sources")
                if sources:
                    for i, s in enumerate(sources, 1):
                        with st.expander(f"{i}. {s['file']} (p. {s['page']})"):
                            st.write(f"Relevance score: {s['score']:.3f}")
                else:
                    st.write("No sources found.")
                    
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                st.write("Full error:", e)
