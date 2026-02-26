from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None

    if request.method == "POST":
        review = request.form["review"]
        vec = vectorizer.transform([review])

        prob = model.predict_proba(vec)[0][1]
        confidence = round(prob * 100, 2)
        prediction = "Positive" if prob >= 0.5 else "Negative"

    return render_template("index.html", prediction=prediction, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
