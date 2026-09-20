from meeting_pipeline import process_sentence
from decision_extractor import extract_decision


def generate_report(transcript):

    sentences = [
        sentence.strip()
        for sentence in transcript.split(".")
        if sentence.strip()
    ]

    report = {
        "action_items": [],
        "suggestions": [],
        "decisions": [],
        "information": []
    }

    for sentence in sentences:

        result = process_sentence(sentence)

        decision = extract_decision(sentence)

        # Decisions get their own category
        if decision:
            report["decisions"].append(decision)

        elif result["category"] == "action_item":
            report["action_items"].extend(result["action_items"])

        elif result["category"] == "suggestion":
            report["suggestions"].append(sentence)

        else:
            report["information"].append(sentence)

    return report


if __name__ == "__main__":

    transcript = """
    Rahul will prepare the presentation by Friday.
    I think we should use MongoDB for the backend.
    We decided to use Flask for the API.
    The meeting is scheduled for Monday.
    """

    report = generate_report(transcript)

    for category, items in report.items():

        print(f"\n{category.upper()}:")

        for item in items:
            print("-", item)