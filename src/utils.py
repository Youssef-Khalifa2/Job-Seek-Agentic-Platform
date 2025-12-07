import re

def filter_reasoning(response):
    """
    Filter out reasoning/thinking content from model response.
    
    Handles multiple formats:
    1. Content blocks with 'type' field (e.g., 'thinking' vs 'text')
    2. XML-style tags like <thinking>...</thinking> or <reasoning>...</reasoning>
    3. String content with embedded reasoning markers
    """
    
    # If response.content is a list of content blocks
    if isinstance(response.content, list):
        filtered_parts = []
        for block in response.content:
            # Handle dict-style blocks
            if isinstance(block, dict):
                block_type = block.get("type", "")
                # Skip reasoning/thinking blocks
                if block_type in ("thinking", "reasoning", "thought"):
                    continue
                # Keep text blocks
                if block_type == "text":
                    filtered_parts.append(block.get("text", ""))
                else:
                    # Keep other block types as-is
                    filtered_parts.append(str(block))
            # Handle string parts
            elif isinstance(block, str):
                filtered_parts.append(block)
        
        return "\n".join(filtered_parts).strip()
    
    # If response.content is a string, filter out XML-style reasoning tags
    if isinstance(response.content, str):
        content = response.content
        
        # Remove common reasoning tag patterns
        patterns = [
            r"<thinking>.*?</thinking>",
            r"<reasoning>.*?</reasoning>",
            r"<thought>.*?</thought>",
            r"<scratchpad>.*?</scratchpad>",
        ]
        
        for pattern in patterns:
            content = re.sub(pattern, "", content, flags=re.DOTALL | re.IGNORECASE)
        
        return content.strip()
    
    # Fallback: return as-is
    return response.content