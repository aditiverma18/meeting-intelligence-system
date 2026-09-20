import spacy

nlp = spacy.load("en_core_web_sm")

ACTION_CUES = {
    "will",
    "shall",
    "should",
    "must"
}


def extract_action_items(text):

    doc = nlp(text)

    # Is this sentence actually assigning future work?
    has_action_cue = any(
        token.text.lower() in ACTION_CUES
        for token in doc
    )

    is_suggestion = False  # Initialize the suggestion flag

    if not has_action_cue:
        return []

    result = []

    for token in doc:

        if token.pos_ != "VERB":
            continue

        person = None
        objects = []

        for child in token.children:

            if child.dep_ in ("nsubj", "nsubjpass"):
                person = child.text

            elif child.dep_ in ("dobj", "obj", "pobj"):
                objects.append(child.text)

        deadline = None

        for ent in doc.ents:
            if ent.label_ in ("DATE", "TIME"):
                deadline = ent.text

        if person and objects:

            result.append({
                "person": person,
                "action": f"{token.lemma_} {' '.join(objects)}",
                "deadline": deadline
            })

    return result