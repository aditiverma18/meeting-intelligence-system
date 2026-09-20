from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------
# 1. LOAD MODEL
# -----------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------------------
# 2. OUR SENTENCES
# -----------------------------------------

sentences = [
    "We should move the meeting to Friday.",
    "Let's reschedule our meeting for Friday.",
    "The prototype needs three buttons.",
    "I like pizza."
]


# -----------------------------------------
# 3. CREATE EMBEDDINGS
# -----------------------------------------

embeddings = model.encode(sentences)

print("Embedding shape:")
print(embeddings.shape)


# -----------------------------------------
# 4. COMPARE SENTENCES
# -----------------------------------------

similarity = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)

print("\nSimilarity between sentence 1 and 2:")
print(similarity[0][0])


similarity = cosine_similarity(
    [embeddings[0]],
    [embeddings[2]]
)

print("\nSimilarity between sentence 1 and 3:")
print(similarity[0][0])


similarity = cosine_similarity(
    [embeddings[0]],
    [embeddings[3]]
)

print("\nSimilarity between sentence 1 and 4:")
print(similarity[0][0])