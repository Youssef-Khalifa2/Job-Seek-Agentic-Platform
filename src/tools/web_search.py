from langchain_community.tools import DuckDuckGoSearchRun
from src.llm_registry import LLMRegistry
from typing import List, Dict, Any
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class JobMarketResearch:
    """
    Analyze job descriptions and validate requirements with web search.
    """

    def __init__(self):
        self.search = DuckDuckGoSearchRun()
        # Nova Lite is perfect here: fast, cheap, and good at extraction tasks
        self.llm = LLMRegistry.get_nova_lite()

    def extract_requirements_from_jd(self, job_description: str) -> Dict[str, Any]:
        """
        Use LLM to parse entire JD and extract structured requirements.
        Handles experience-based skills like "5 years managing Snowflake".
        """
        extraction_prompt = f"""
        Analyze this job description and extract ALL requirements in detail.

        Job Description:
        {job_description}

        Extract:
        1. **Job Title**: The specific role
        2. **Required Skills**: Technical skills, tools, languages.
        3. **Experience Requirements**: Years of experience + context.
        4. **Preferred Skills**: Nice-to-have skills.
        5. **Domain Expertise**: Industry knowledge (e.g., fintech).

        Return JSON:
        {{
          "job_title": "...",
          "required_skills": ["skill1", "skill2"],
          "experience_requirements": [
            {{"years": 5, "skill": "Snowflake", "context": "warehouse management"}}
          ],
          "preferred_skills": ["skill1"],
          "domain_expertise": ["domain1"]
        }}
        """

        try:
            response = self.llm.invoke(extraction_prompt)
            # Basic cleaning if the model adds markdown
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]
            return json.loads(content)
        except Exception as e:
            print(f"⚠️ JD Parsing failed: {e}")
            return {"error": "Failed to parse JD"}

    def validate_skill_requirement(self, skill: str, job_title: str) -> Dict[str, Any]:
        """
        Check if a skill is commonly required for a job title via web search.
        Useful for checking if a skill is a 'hard requirement' or just a keyword.
        """
        query = f'"{job_title}" job requirements "{skill}"'

        try:
            # Use ThreadPoolExecutor with timeout to prevent hanging
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.search.run, query)
                results = future.result(timeout=10)  # 10 second timeout

            # Simple heuristic: how often is the skill mentioned in the snippets?
            skill_mentions = results.lower().count(skill.lower())

            return {
                "skill": skill,
                "validated": skill_mentions > 0,
                "mentions": skill_mentions,
                "evidence_snippet": results[:200] + "..."
            }
        except TimeoutError:
            print(f"  ⏱️ Web search timeout for skill: {skill}")
            return {
                "skill": skill,
                "validated": False,
                "error": "Search timeout (10s)",
                "mentions": 0
            }
        except Exception as e:
            return {"skill": skill, "validated": False, "error": str(e)}

# Helper functions
def analyze_job_description(jd: str) -> Dict[str, Any]:
    researcher = JobMarketResearch()
    return researcher.extract_requirements_from_jd(jd)

def validate_market_demand(skill: str, job_title: str) -> Dict[str, Any]:
    researcher = JobMarketResearch()
    return researcher.validate_skill_requirement(skill, job_title)