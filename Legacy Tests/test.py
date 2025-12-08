from src.llm_registry import LLMRegistry
from typing import List
import base64
from io import BytesIO
import pdf2image
from langchain_core.messages import HumanMessage

vlm = LLMRegistry.get_pixtral_large()

def pdf_to_images(pdf_path: str) -> List[str]:
    """Convert all PDF pages to base64-encoded images."""
    images = pdf2image.convert_from_path(pdf_path)

    base64_images = []
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        base64_images.append(base64.b64encode(buffer.read()).decode('utf-8'))

    return base64_images


def create_vision_message(base64_images: List[str]) -> HumanMessage:
    """Create a properly formatted message with text and images."""
    content = [{
                      "type": "text",
                      "text": """
                      Analyze this resume's ACTUAL visual layout for ATS compatibility.

                      Check for:
                      1. Multiple columns (ATS can't parse columns)
                      2. Tables or grids
                      3. Icons or graphics
                      4. Unusual fonts or formatting
                      5. Text boxes or sidebars

                      Return JSON: {...}
                      """
                  }]
    
    for img_b64 in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })
    
    return HumanMessage(content=content)


pdf_path = r"F:\ML Projects\Job-Seek-platform\Job-Seek-Agentic-Platform\cvs\Youssef Khalifa CV.pdf"
base64_images = pdf_to_images(pdf_path)

message = create_vision_message(base64_images)
response = vlm.invoke([message])

print(response.content)