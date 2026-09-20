from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def summarize_meeting(transcript):
    inputs = tokenizer(
        transcript,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    output = model.generate(
        **inputs,
        max_length=120,
        min_length=30,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


if __name__ == "__main__":

    transcript = """
    The team discussed the backend architecture.
    Rahul suggested using MongoDB.
    Aditi suggested using Flask for the API.
    The team decided to use Flask and MongoDB.
    Rahul will prepare the presentation by Friday.
    """

    print("\nSUMMARY:")
    print(summarize_meeting(transcript))