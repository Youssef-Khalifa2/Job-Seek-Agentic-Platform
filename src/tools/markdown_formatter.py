from typing import Dict, Any
import markdown
from weasyprint import HTML
import os

class MarkdownFormatter:
    """
    Format resume as markdown and export to PDF.
    """

    @staticmethod
    def format_resume_as_markdown(resume_text: str) -> str:
        """
        Ensure resume text is properly formatted as markdown.
        Adds proper headers, bold text, bullet points, etc.
        """
        # Resume should already be markdown from editor
        # This function ensures consistent formatting

        lines = resume_text.split('\n')
        formatted_lines = []

        for line in lines:
            stripped = line.strip()

            # Ensure headers have proper markdown syntax
            if stripped and not stripped.startswith('#') and len(stripped) < 50:
                # Might be a section header - check if all caps or title case
                if stripped.isupper() or stripped.istitle():
                    formatted_lines.append(f"## {stripped}")
                    continue

            formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    @staticmethod
    def markdown_to_pdf(markdown_text: str, output_path: str) -> str:
        """
        Convert markdown to PDF using WeasyPrint.
        """
        # Convert markdown to HTML
        html_content = markdown.markdown(
            markdown_text,
            extensions=['extra', 'nl2br', 'sane_lists']
        )

        # Add CSS styling for professional resume look
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Helvetica', 'Arial', sans-serif;
                    font-size: 11pt;
                    line-height: 1.4;
                    margin: 0.5in;
                    color: #333;
                }}
                h1 {{
                    font-size: 20pt;
                    margin-bottom: 0.2em;
                    color: #2c3e50;
                    border-bottom: 2px solid #2c3e50;
                }}
                h2 {{
                    font-size: 14pt;
                    margin-top: 1em;
                    margin-bottom: 0.3em;
                    text-transform: uppercase;
                    color: #2c3e50;
                    border-bottom: 1px solid #eee;
                }}
                h3 {{
                    font-size: 12pt;
                    margin-top: 0.5em;
                    margin-bottom: 0.2em;
                    font-weight: bold;
                }}
                ul {{
                    margin-top: 0.3em;
                    margin-bottom: 0.5em;
                    padding-left: 1.5em;
                }}
                li {{
                    margin-bottom: 0.2em;
                }}
                strong {{
                    color: #000;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Generate PDF
        HTML(string=styled_html).write_pdf(output_path)
        return output_path

# Helper functions
def export_resume_as_pdf(resume_text: str, output_path: str) -> str:
    """Convert resume to PDF."""
    formatter = MarkdownFormatter()
    formatted = formatter.format_resume_as_markdown(resume_text)
    return formatter.markdown_to_pdf(formatted, output_path)