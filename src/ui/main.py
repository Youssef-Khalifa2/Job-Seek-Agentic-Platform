import streamlit as st
# from .components import render_sidebar

def main():
    st.set_page_config(page_title="Career Nexus", layout="wide")
    
    st.title("Career Nexus")
    
    # mode = render_sidebar()
    
    # if mode == "Job Seeker":
    #     render_job_seeker_view()
    # else:
    #     render_recruiter_view()

if __name__ == "__main__":
    main()
