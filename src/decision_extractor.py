import re


DECISION_PATTERNS = [
    r"\bwe decided to (.+)",
    r"\bwe have decided to (.+)",
    r"\bthe team decided to (.+)",
    r"\bwe agreed to (.+)",
    r"\bthe team agreed to (.+)",
    r"\blet's go with (.+)",
    r"\blet's use (.+)",
    r"\bwe'll use (.+)",
]


def extract_decision(text):
    text_lower = text.lower().strip()

    for pattern in DECISION_PATTERNS:
        match = re.search(pattern, text_lower)

        if match:
            return match.group(1).strip()

    return None


if __name__ == "__main__":
    examples = [
        "We decided to use MongoDB.",
        "The team agreed to use Flask.",
        "Let's go with Python.",
        "The meeting is scheduled for Monday."
    ]

    for text in examples:
        print(text, "->", extract_decision(text))