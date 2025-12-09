import streamlit as st
import os
import sys
import tempfile

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.JobSeeker.graph import build_job_seeker_graph 
from src.ingestion.pipeline import ingest_cv, extract_text_from_pdf
from src.tools.markdown_formatter import export_resume_as_pdf

def render_sidebar():
    return st.sidebar.radio("Mode", ["Job Seeker (Agent)", "Recruiter (Chat)"])

def render_job_seeker_view():
    st.subheader("🚀 CV Optimizer")
    
    # 1. File Upload
    uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])
    
    # 2. Job Description Input
    job_description = st.text_area("Paste the Job Description here:", height=150)

    # 3. The "Go" Button
    if st.button("Optimize My CV"):
        if uploaded_file and job_description:
            with st.spinner("Processing your CV..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_pdf_path = tmp_file.name
                
                ingest_cv(temp_pdf_path) 
                
                resume_text = extract_text_from_pdf(temp_pdf_path)
                
                initial_state = {
                    "resume_text": resume_text,
                    "job_description": job_description,
                    "original_pdf_path": temp_pdf_path,  # ADD THIS - needed for ATS critic
                    "revision_count": 0,
                    "critique_inputs": [],
                    "actionable_critiques": [],
                    "messages": []  # ADD THIS - required by state schema
                }
                
                st.info("🤖 Agent is analyzing and optimizing...")
                
                graph = build_job_seeker_graph()
                final_state = graph.invoke(initial_state)
                
                st.success("Optimization Complete!")
                output_pdf_path = "optimized_cv.pdf"
                try:
                    export_resume_as_pdf(final_state["resume_text"], output_pdf_path)
                    
                    with open(output_pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="📄 Download Optimized PDF",
                            data=pdf_file,
                            file_name="Optimized_CV.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"Could not generate PDF: {e}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📝 Original CV")
                    st.text(resume_text) # Preview
                with col2:
                    st.markdown("### ✨ Optimized CV")
                    st.text(final_state.get("resume_text",None))
                    
                # Cleanup temp file
                os.remove(temp_pdf_path)
        else:
            st.warning("Please upload a CV and paste a Job Description.")

def render_recruiter_view():
    st.subheader("🔍 Recruiter RAG Agent")
    # You can import your run_rag_pipeline here later
    st.info("Chat functionality coming up next!")

def main():
    st.set_page_config(page_title="Career Nexus", layout="wide")
    st.title("Career Nexus")
    
    mode = render_sidebar()
    
    if mode == "Job Seeker (Agent)":
        render_job_seeker_view()
    else:
        render_recruiter_view()

if __name__ == "__main__":
    main()