# File: src/tools/grammar_check.py
import language_tool_python
from typing import Dict, Any

class GrammarChecker:
    """
    Wraps LanguageTool to provide objective grammar and spelling checks.
    This acts as the 'Ground Truth' for the Language Critic.
    """

    def __init__(self):
        self.tool = language_tool_python.LanguageTool('en-US')

    def check_text(self, text: str) -> Dict[str, Any]:
        """
        Scans text for grammar issues.
        """
        matches = self.tool.check(text)
        
        # We limit to top 15 issues to avoid overwhelming the LLM context
        issues = []
        for match in matches[:15]:
            issues.append({
                "message": match.message,
                "context": match.context,
                "replacements": match.replacements[:3], # Top 3 suggestions
                "rule_id": match.ruleId,
                "category": match.category
            })

        return {
            "issue_count": len(matches),
            "critical_issues": issues,
            "tool_status": "success"
        }

# Helper
def check_grammar(text: str) -> Dict[str, Any]:
    checker = GrammarChecker()
    return checker.check_text(text)