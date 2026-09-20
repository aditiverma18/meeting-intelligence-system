from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


sentences = [
    "Rahul will finish the API integration by Friday.",
    "Aditi will send the report tomorrow.",
    "We decided to use MongoDB for the backend.",
    "Let's use Flask for the API.",
    "Do we have the API credentials?",
    "When will the deployment happen?",
    "The API currently has three endpoints.",
    "The backend is running on Flask."
]

labels = [
    "ACTION",
    "ACTION",
    "DECISION",
    "DECISION",
    "QUESTION",
    "QUESTION",
    "INFORMATION",
    "INFORMATION"
]


# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(sentences)


# Train classifier
model = LogisticRegression()

model.fit(X, labels)


# Test with a new sentence
new_sentence = [
    "Rahul will complete the database work tomorrow."
]

new_X = vectorizer.transform(new_sentence)

prediction = model.predict(new_X)

print("Prediction:", prediction[0])