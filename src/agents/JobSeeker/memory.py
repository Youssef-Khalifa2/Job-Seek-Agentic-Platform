# File: src/agents/JobSeeker/memory.py
from typing import List, Dict, Any
from datetime import datetime
import json

class RevisionMemory:
    """
    Tracks what was tried, what worked, and what failed.
    This is the 'Meta-Cognition' layer for the agent.
    """

    @staticmethod
    def log_revision(
        iteration: int,
        critiques: List[Any],  # Changed from critiques_applied to match caller
        quality_before: float,
        quality_after: float,
        success: bool
    ) -> Dict[str, Any]:
        """
        Creates a structured log entry for a single revision loop.
        """
        # Calculate impact
        quality_delta = quality_after - quality_before

        entry = {
            "iteration": iteration,
            "timestamp": str(datetime.now()),
            "critiques_used": [c.source for c in critiques],  # Updated variable name
            "quality_score_before": quality_before,
            "quality_score_after": quality_after,
            "improvement_delta": round(quality_delta, 2),
            "verification_passed": success
        }
        
        return entry

    @staticmethod
    def analyze_effectiveness(history: List[Dict[str, Any]]) -> str:
        """
        Generates a readable report on which agents are pulling their weight.
        """
        if not history:
            return "No revision history available."

        report = "### 🧠 Agent Effectiveness Report\n"
        
        # Simple aggregation
        total_improvement = 0
        critic_counts = {}
        
        for entry in history:
            total_improvement += entry["improvement_delta"]
            for critic in entry["critiques_used"]:
                critic_counts[critic] = critic_counts.get(critic, 0) + 1

        report += f"- **Total Improvement:** {total_improvement:+.1f} points\n"
        report += f"- **Iterations:** {len(history)}\n"
        report += "- **Active Critics:**\n"
        
        for critic, count in critic_counts.items():
            report += f"  - {critic}: participated in {count} loops\n"

        return report

# Helper function to dump the report to disk
def save_memory_report(history: List[Dict[str, Any]], filename: str = "agent_memory_report.md"):
    report = RevisionMemory.analyze_effectiveness(history)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  💾 Memory report saved to {filename}")