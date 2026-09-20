import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "aditiverma6/meeting-dialogue-act-bert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()


def predict_dialogue_act(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    return model.config.id2label[prediction]


if __name__ == "__main__":
    text = input("Enter a sentence: ")
    print("Dialogue Act:", predict_dialogue_act(text))