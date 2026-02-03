# utils.py

def safe_system_prompt(prompt: str) -> str:
    """
    This function is intended to sanitize or validate system prompts.
    For now, it just returns the input unchanged.
    You can later add more safety checks if needed.
    """
    return prompt

def truncate_messages(messages, max_tokens=4096, completion_tokens=500):
    """
    Roughly trims conversation messages to stay within model context limits.
    This is character-based approximation (safe + fast).
    """

    if not messages:
        return messages

    # Reserve space for completion
    max_input_tokens = max_tokens - completion_tokens

    # Rough token estimate: 1 token ≈ 4 characters
    max_chars = max_input_tokens * 4

    total_chars = 0
    trimmed = []

    # Walk backwards to preserve most recent context
    for msg in reversed(messages):
        content = msg.get("content", "")
        total_chars += len(content)

        if total_chars > max_chars:
            break

        trimmed.insert(0, msg)

    return trimmed
