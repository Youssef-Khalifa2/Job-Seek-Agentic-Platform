# File: src/tools/grammar_check.py
import language_tool_python
from typing import Dict, Any

class GrammarChecker:
    """
    Wraps LanguageTool to provide objective grammar and spelling checks.
    This acts as the 'Ground Truth' for the Language Critic.
    """

    def __init__(self):
        try:
            self.tool = language_tool_python.LanguageTool('en-US')
            self.enabled = True
        except Exception as e:
            print(f"⚠️ Grammar tool initialization failed: {e}")
            print("   Language Critic will run without grammar checking")
            self.tool = None
            self.enabled = False

    def check_text(self, text: str) -> Dict[str, Any]:
        """
        Scans text for grammar issues.
        """
        # Check if tool is available
        if not self.enabled or not self.tool:
            return {
                "issue_count": 0,
                "critical_issues": [],
                "tool_status": "disabled",
                "error": "Grammar tool not available"
            }

        try:
            matches = self.tool.check(text)

            # We limit to top 15 issues to avoid overwhelming the LLM context
            issues = []
            for match in matches[:15]:
                issues.append({
                    "message": getattr(match, 'message', ''),
                    "context": getattr(match, 'context', ''),
                    "replacements": getattr(match, 'replacements', [])[:3],  # Top 3 suggestions
                    "rule_id": getattr(match, 'ruleId', getattr(match, 'rule_id', '')),
                    "category": getattr(match, 'category', '')
                })

            return {
                "issue_count": len(matches),
                "critical_issues": issues,
                "tool_status": "success"
            }
        except Exception as e:
            print(f"⚠️ Grammar check failed: {e}")
            return {
                "issue_count": 0,
                "critical_issues": [],
                "tool_status": "error",
                "error": str(e)
            }

# Helper
def check_grammar(text: str) -> Dict[str, Any]:
    checker = GrammarChecker()
    return checker.check_text(text)