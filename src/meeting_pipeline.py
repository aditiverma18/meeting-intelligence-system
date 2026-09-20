from predict import predict_dialogue_act
from action_items import extract_action_items

def process_sentence(text):
    label = predict_dialogue_act(text)
    actions = extract_action_items(text)

    suggestion_cues = [
    "suggested",
    "suggest",
    "proposed",
    "recommend",
    "recommended",
    "i think we should",
    "we should"
]

    is_suggestion = (
    label == "sug"
    or any(cue in text.lower() for cue in suggestion_cues)
)
    # Suggestions are not automatically action items
    if label == "sug":
        category = "suggestion"
    elif actions:
        category = "action_item"
    else:
        category = "information"

    return {
        "text": text,
        "dialogue_act": label,
        "category": category,
        "action_items": actions if category == "action_item" else []
    }


if __name__ == "__main__":
    sentences = [
        "Rahul will prepare the presentation by Friday.",
        "I think we should use MongoDB for the backend.",
        "The meeting is scheduled for Monday."
    ]

    for sentence in sentences:
        print(process_sentence(sentence))