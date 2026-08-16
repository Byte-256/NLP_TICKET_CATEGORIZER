import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

def load_dummy_data():
    data = {
        'text': [
            # Billing
            "Need help with my invoice", "Credit card was charged twice", "Update my billing address",
            "Payment failed to process", "I was overcharged on my statement", "Monthly statement is incorrect",
            "Where is my invoice?", "Billing issue with credit card", "Refund delayed",

            # Technical
            "Cannot connect to the VPN", "Server is down since morning", "Laptop screen is completely black",
            "Password reset required", "My Wi-Fi keeps disconnecting", "Software crashes constantly",
            "Cannot export PDF files", "Wi-Fi and internet is down", "API returns 500 error",

            # HR
            "How do I request PTO?", "Need to update my direct deposit", "Where is the employee handbook?",
            "Maternity leave policy", "Add newborn to health insurance", "Health insurance enrollment questions",
            "Direct deposit is late", "PTO request denied",

            # General
            "What time is the all-hands meeting?", "Is the cafeteria open today?", "Parking pass renewal",
            "Lost my ID badge", "Where is the guest parking located?", "Guest parking pass needed",
            "Cafeteria menu for today", "All-hands meeting location"
        ],
        'category': [
            "Billing", "Billing", "Billing", "Billing", "Billing", "Billing", "Billing", "Billing", "Billing",
            "Technical", "Technical", "Technical", "Technical", "Technical", "Technical", "Technical", "Technical", "Technical",
            "HR", "HR", "HR", "HR", "HR", "HR", "HR", "HR",
            "General", "General", "General", "General", "General", "General", "General", "General"
        ]
    }
    return pd.DataFrame(data)


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Only keep lowercase letters and spaces
    return text


def assign_proirity(text):
    urgent_keywords = ['urgent', 'down', 'crashes', 'not working', 'emergency', 'asap', 'critical']
    text = clean_text(text).split(" ")
    if any(keyword in text for keyword in urgent_keywords):
        return "URGENT"
    return "NORMAL"


def print_output(text, pred, probablity):
    proirity = assign_proirity(text)
    print(f"Text: {text}, Predicted Category: {pred}, Probability: {probablity * 100:.2f}%, Human Review Needed: {'YES' if probablity < 0.6 else 'NO'}, Priority: {proirity}")


def interactive_CLI(model, vectorizer):
    while True:
        print("Interactive CLI (type 'exit' to quit):")
        text = input("Enter text to ticket: ")
        if text.lower() == 'exit':
            break
        text = clean_text(text)
        textV = vectorizer.transform([text])
        pred = model.predict(textV)
        probablity = model.predict_proba(textV).max()
        print_output(text, pred[0], probablity)


def main():
    print("Hello from nlt-token-categorizer!")
    df = load_dummy_data()

    df['cleaned_text'] = df['text'].apply(clean_text)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['cleaned_text'], df['category'], test_size=0.2, random_state=42)

    #Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    #Training
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    #Prediction
    nb_predictions = nb_model.predict(X_test)

    #Evaluation
    nb_accuracy = accuracy_score(y_test, nb_predictions)
    print(f"Naive Bayes Accuracy = {nb_accuracy * 100:.2f} %")

    print("---Sample Test Ticket---")

    Sample_Test_Ticket = ["Wifi is ok, server can't reached",
        "Need help with invoice.",
        "procedure for taking a leave >``<",
        "where i can find past bills?",
        "my pass isn't valid"]
    clean_sample = [clean_text(Sample_Test_Ticket_text) for Sample_Test_Ticket_text in Sample_Test_Ticket]

    new_test = vectorizer.transform(clean_sample)
    new_pred = nb_model.predict(new_test)
    new_pred_proba = nb_model.predict_proba(new_test)

    for text, pred, probablity in zip(clean_sample, new_pred, new_pred_proba):
        print_output(text, pred, probablity.max())

    print("---User can now interact with the model and get prediction---")
    interactive_CLI(nb_model, vectorizer)

if __name__ == "__main__":
    main()
