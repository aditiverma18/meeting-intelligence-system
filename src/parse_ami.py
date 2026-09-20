from pathlib import Path
import re
import xml.etree.ElementTree as ET
import pandas as pd

AMI_DIR = Path("data/ami_public_manual_1.6.2")

NITE = "http://nite.sourceforge.net/"

ONTOLOGY_FILE = AMI_DIR / "ontologies" / "da-types.xml"

ontology_root = ET.parse(ONTOLOGY_FILE).getroot()

dialogue_labels = {}

for element in ontology_root.iter():

    da_id = element.attrib.get(f"{{{NITE}}}id")
    name = element.attrib.get("name")

    if da_id and da_id.startswith("ami_da_"):
        dialogue_labels[da_id] = name


# --------------------------------------------------
# 2. EXTRACT WORD IDS FROM A SPAN
# --------------------------------------------------

def extract_word_ids(href):

    pattern = r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?"

    match = re.search(pattern, href)

    if not match:
        return []

    start_id = match.group(1)
    end_id = match.group(2)

    # Single word
    if end_id is None:
        return [start_id]

    start_match = re.search(r"(\d+)$", start_id)
    end_match = re.search(r"(\d+)$", end_id)

    if not start_match or not end_match:
        return []

    start_num = int(start_match.group(1))
    end_num = int(end_match.group(1))

    prefix = start_id[:start_match.start()]

    return [
        f"{prefix}{i}"
        for i in range(start_num, end_num + 1)
    ]


# --------------------------------------------------
# 3. CLEAN TEXT
# --------------------------------------------------

def clean_text(text):

    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+'\s*", "'", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# 4. PARSE ONE DIALOGUE FILE
# --------------------------------------------------

def parse_dialogue_file(dialogue_file):

    # Find corresponding words file
    words_file = AMI_DIR / "words" / dialogue_file.name.replace(
        ".dialog-act.xml",
        ".words.xml"
    )

    if not words_file.exists():
        return []

    dialogue_root = ET.parse(dialogue_file).getroot()
    words_root = ET.parse(words_file).getroot()

    # Build word dictionary
    words = {}

    for element in words_root.iter():

        if element.tag.endswith("w"):

            word_id = element.attrib.get(
                f"{{{NITE}}}id"
            )

            if word_id:
                words[word_id] = element.text or ""

    # Extract speaker
    match = re.search(
        r"\.([A-Z])\.dialog-act\.xml$",
        dialogue_file.name
    )

    speaker = match.group(1) if match else "UNKNOWN"

    # Extract meeting
    meeting = dialogue_file.name.split(".")[0]

    examples = []

    for dact in dialogue_root:

        label_id = None
        word_href = None

        for element in dact:

            if element.tag.endswith("pointer"):

                href = element.attrib.get("href", "")

                match = re.search(
                    r"id\((ami_da_\d+)\)",
                    href
                )

                if match:
                    label_id = match.group(1)

            elif element.tag.endswith("child"):

                word_href = element.attrib.get("href", "")

        if not label_id or not word_href:
            continue

        word_ids = extract_word_ids(word_href)

        text_words = [
            words[word_id]
            for word_id in word_ids
            if word_id in words
        ]

        text = clean_text(" ".join(text_words))

        if text:

            examples.append({
                "meeting": meeting,
                "speaker": speaker,
                "text": text,
                "label_id": label_id,
                "label": dialogue_labels.get(
                    label_id,
                    "UNKNOWN"
                )
            })

    return examples


# --------------------------------------------------
# 5. FIND ALL DIALOGUE-ACT FILES
# --------------------------------------------------

dialogue_files = list(
    (AMI_DIR / "dialogueActs").glob("*.dialog-act.xml")
)

print("Dialogue-act files found:", len(dialogue_files))


# --------------------------------------------------
# 6. PARSE EVERYTHING
# --------------------------------------------------

dataset = []

for i, dialogue_file in enumerate(dialogue_files):

    examples = parse_dialogue_file(dialogue_file)

    dataset.extend(examples)

    if (i + 1) % 50 == 0:
        print(
            f"Processed {i + 1}/{len(dialogue_files)} files"
        )


# --------------------------------------------------
# 7. CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(dataset)


# --------------------------------------------------
# 8. BASIC DATASET INFORMATION
# --------------------------------------------------

print("\n==============================")
print("DATASET SUMMARY")
print("==============================")

print("Total examples:", len(df))
print("Number of classes:", df["label"].nunique())

print("\nClass distribution:")
print(df["label"].value_counts())


# --------------------------------------------------
# 9. SHOW EXAMPLES
# --------------------------------------------------

print("\nFirst 10 examples:")

print(
    df[
        ["meeting", "speaker", "text", "label"]
    ].head(10).to_string(index=False)
)


# --------------------------------------------------
# 10. SAVE DATASET
# --------------------------------------------------

output_file = "data/ami_dialogue_dataset.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nDataset saved to:", output_file)