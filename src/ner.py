import spacy

nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    doc = nlp(text)

    entities = []

    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })

    return entities


if __name__ == "__main__":
    text = "Rahul will send the report to Aditi on Friday."

    print(extract_entities(text))