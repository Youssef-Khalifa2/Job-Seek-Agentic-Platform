# File: src/tools/ats_parser.py
import os
import spacy
# Note: In a real env, you'd need: pip install pyresparser nltk spacy
# and: python -m spacy download en_core_web_sm
from pyresparser import ResumeParser
from typing import Dict, Any

class ATSParser:
    """
    Wraps an off-the-shelf ATS library (pyresparser) to simulate
    real-world parsing behavior (The 'Black Box').
    """

    @staticmethod
    def parse_resume(pdf_path: str) -> Dict[str, Any]:
        """
        Runs the resume through a standard extraction library.
        Returns exactly what an ATS would 'see' (or miss).
        """
        try:
            # ResumeParser extracts skills, education, email, etc.
            data = ResumeParser(pdf_path).get_extracted_data()
            
            # We explicitly check for 'parsing failures' to report to the Agent
            missing_fields = []
            if not data.get("email"): missing_fields.append("email")
            if not data.get("mobile_number"): missing_fields.append("phone")
            if not data.get("skills"): missing_fields.append("skills")
            if not data.get("total_experience"): missing_fields.append("experience")

            return {
                "ats_output": data,
                "missing_critical_fields": missing_fields,
                "parsable_status": len(missing_fields) == 0,
                "engine": "pyresparser"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "parsable_status": False,
                "engine": "pyresparser"
            }

# Helper for the Agent
def check_ats_compatibility(pdf_path: str) -> Dict[str, Any]:
    return ATSParser.parse_resume(pdf_path)