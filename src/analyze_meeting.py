from meeting_report import generate_report
from summarizer import summarize_meeting


def analyze_meeting(transcript):

    report = generate_report(transcript)

    summary = summarize_meeting(transcript)

    report["summary"] = summary

    return report


if __name__ == "__main__":

    transcript = """
    Rahul will prepare the presentation by Friday.
    I think we should use MongoDB for the backend.
    We decided to use Flask for the API.
    The meeting is scheduled for Monday.
    """

    report = analyze_meeting(transcript)

    print("\n===== MEETING SUMMARY =====")
    print(report["summary"])

    print("\n===== ACTION ITEMS =====")
    for item in report["action_items"]:
        print(item)

    print("\n===== SUGGESTIONS =====")
    for item in report["suggestions"]:
        print(item)

    print("\n===== DECISIONS =====")
    for item in report["decisions"]:
        print(item)

    print("\n===== INFORMATION =====")
    for item in report["information"]:
        print(item)