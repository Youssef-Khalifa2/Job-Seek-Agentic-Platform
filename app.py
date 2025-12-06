import streamlit as st
from LLM import run_rag_pipeline
# Page Config
st.set_page_config(
    page_title="CV Chatbot",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("📄 CV Chatbot")
    st.markdown("---")
    st.markdown("This AI assistant helps you find the best candidates based on their CVs.")
    
    if st.button("Clear Chat", type="primary"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This system uses RAG (Retrieval Augmented Generation) "
        "to search through candidate CVs and provide relevant answers."
    )

# Main Chat Interface
st.title("CV Chatbot")
st.caption("Ask questions about candidate experience, skills, and more.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "evidence" in message:
            with st.expander("View Evidence"):
                st.markdown(message["evidence"])

# User Input
if prompt := st.chat_input("Ask a question about the candidates..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Searching CVs..."):
            try:
                result = run_rag_pipeline(prompt)
                answer = result['answer']
                evidence = result.get('contexts', 'No specific evidence found.')
                
                # Format evidence if it's a list
                if isinstance(evidence, list):
                    evidence = "\n\n---\n\n".join(evidence)

                st.markdown(answer)
                with st.expander("View Evidence"):
                    st.markdown(evidence)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    
                    "role": "assistant", 
                    "content": answer,
                    "evidence": evidence
                })
            except Exception as e:
                st.error(f"An error occurred: {e}")