import os
import asyncio
from src.agents.JobSeeker.graph import build_job_seeker_graph # Adjust casing if you renamed folder
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

# Load API Keys
load_dotenv()

def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])

SAMPLE_RESUME = extract_text_from_pdf(r"F:\ML Projects\Job-Seek-platform\Job-Seek-Agentic-Platform\cvs\Hassan Mohamed Resume.pdf")

SAMPLE_JOB_DESCRIPTION = """
Letters of Credit Operations Officer
Performing tasks relating to LCs in the most effective and efficient manner resulting in the lowest cost , highest level of quality and ensuring that the related financial and operational risks are mitigated. 
Process import letters of credit in accordance with intl rules and regulations in a way that supports the flow of shipments and acts as a direct interface with suppliers' finance team & Egyptian banks.
Process transactions for issuing LCs, amendment, cancellation, lodgment of claims and payments within the limit as per the credit policy. 
Follow-up with retail & finance team to execute bank transfers to meet supplier's payment on a timely manner as per our agreements. 
Maintain accounts record for suppliers.
Process transactions relating to advising, amendment and cancellation of import letters of credit in compliance with  bank's policies and procedures. 
Input transactions for advising, confirmation, amendment and cancellation of import letters of credit on a timely basis.
Settle value of commercial invoices , ensure full funding coverage with finance to meet supplier's commitments.
Prepare and submit reports to monitor the workflow and payments condition.
"""

def run_test():
    print("🚀 Initializing Job Seeker Agent...")
    
    # 1. Build Graph
    graph = build_job_seeker_graph()
    
    # 2. Define Initial State
    initial_state = {
        "resume_text": SAMPLE_RESUME,
        "job_description": SAMPLE_JOB_DESCRIPTION,
        "revision_count": 0,
        "critique_inputs": [],
        "actionable_critiques": []
    }
    
    # 3. Run (Invoke)
    print("🏃 Running Workflow (this may take a minute)...")
    try:
        final_state = graph.invoke(initial_state)
        
        print("\n✅ WORKFLOW FINISHED!")
        print(f"Total Revisions: {final_state['revision_count']}")
        print("\n--- FINAL OPTIMIZED CV ---\n")
        print(final_state['resume_text'])  # The working resume that gets updated each loop
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


def render_to_png(graph, output_path: str = "graph.png"):
    """
    Render graph to PNG image file using Mermaid.
    No longer requires pygraphviz system package.
    """
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"✓ Graph saved to: {output_path}")
        return True
    except Exception as e:
        print(f"✗ PNG rendering failed: {e}")
        return False

if __name__ == "__main__":
    #run_test()
    render_to_png(build_job_seeker_graph())