import base64
import json
from io import BytesIO
from typing import List, Dict, Any
from pdf2image import convert_from_path
from langchain_core.messages import HumanMessage
from src.llm_registry import LLMRegistry

class LayoutAnalyzer:
    """
    Uses a Vision Language Model (VLM) to "see" the resume layout.
    Detects visual issues that text parsers miss (columns, weird margins, color contrast).
    """

    def __init__(self):
        # We use Pixtral Large (or your chosen VLM)
        self.vlm = LLMRegistry.get_pixtral_large()

    def _pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        Internal helper: Convert PDF pages to base64-encoded images.
        """
        try:
            # Convert first 2 pages only (saves tokens/time)
            images = convert_from_path(pdf_path, first_page=1, last_page=2)
            
            base64_images = []
            for img in images:
                # Resize if needed to save tokens, but usually 150 DPI is fine
                buffer = BytesIO()
                img.save(buffer, format='PNG') 
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode('utf-8')
                base64_images.append(img_str)
                
            return base64_images
        except Exception as e:
            print(f"❌ Error converting PDF to images: {e}")
            return []

    def analyze_layout(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main method called by the Agent.
        """
        images = self._pdf_to_images(pdf_path)

        if not images:
            return {
                "has_columns": False,
                "has_graphics": False,
                "has_tables": False,
                "overall_score": 0,
                "layout_issues": ["Error: Could not convert PDF to images"],
                "recommendation": "Unable to analyze layout",
                "error": "Could not convert PDF to images"
            }

        # Construct the VLM prompt
        content_payload = [
            {
                "type": "text",
                "text": """
                You are an Expert Resume Auditor. Analyze this document's VISUAL LAYOUT.
                
                Identify specific ATS "killers":
                1. Multi-column layouts (columns side-by-side).
                2. Graphics, icons, or progress bars (e.g., skill bars).
                3. Tables or visible grid lines.
                4. Text in headers/footers (often ignored by parsers).
                5. Low contrast or illegible fonts.

                Return STRICT JSON:
                {
                    "has_columns": boolean,
                    "has_graphics": boolean,
                    "has_tables": boolean,
                    "overall_score": 0-100,
                    "layout_issues": ["list", "of", "issues"],
                    "recommendation": "string summary"
                }
                """
            }
        ]

        # Append images to the payload
        for img_b64 in images:
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            })

        message = HumanMessage(content=content_payload)

        try:
            print("  👁️ VLM analyzing visual layout...")
            response = self.vlm.invoke([message])
            
            # Clean and Parse JSON
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]
            
            return json.loads(content)

        except Exception as e:
            print(f"❌ VLM Analysis failed: {e}")
            return {
                "has_columns": False,
                "has_graphics": False,
                "has_tables": False,
                "overall_score": 0,
                "layout_issues": [f"Error: {str(e)}"],
                "recommendation": "Unable to analyze layout",
                "error": str(e)
            }

# Helper for the Agent
def analyze_resume_layout(pdf_path: str) -> Dict[str, Any]:
    analyzer = LayoutAnalyzer()
    return analyzer.analyze_layout(pdf_path)