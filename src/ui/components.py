import streamlit as st

def render_sidebar():
    return st.sidebar.radio("Mode", ["Job Seeker", "Recruiter"])
