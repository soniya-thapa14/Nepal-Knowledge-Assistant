def detect_topic(query: str) -> str:
    """
    Detect the topic of the user's query to select relevant online sources.

    Args:
        query (str): User's question

    Returns:
        str: topic key to use in ONLINE_SOURCES
    """
    q = query.lower()

    if any(w in q for w in ["trek", "trekking", "hiking"]):
        return "trekking"
    if any(w in q for w in ["everest", "mountain", "climb", "climbing"]):
        return "mountaineering"
    if any(w in q for w in ["visa", "permit", "immigration"]):
        return "visa"
    if any(w in q for w in ["festival", "culture", "tradition", "heritage"]):
        return "culture"

    # Default topic if nothing matches
    return None
