from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load Dataset
gold = pd.read_excel("gold_price.xlsx")

gold["Date"] = pd.to_datetime(gold["Date"])
gold["year"] = gold["Date"].dt.year
gold["month"] = gold["Date"].dt.month

X = gold[["year", "month"]]
y = gold["Price"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

MIN_YEAR = int(gold["year"].min())

@app.route("/")
def home():
    return render_template(
        "index.html",
        min_year=MIN_YEAR,
        prediction_text=""
    )

@app.route("/predict", methods=["POST"])
def predict():

    year = int(request.form["year"])
    month = int(request.form["month"])

    if year < MIN_YEAR:
        return render_template(
            "index.html",
            min_year=MIN_YEAR,
            prediction_text=f"⚠️ Enter year from {MIN_YEAR} onwards"
        )

    prediction = model.predict([[year, month]])

    result = f"💰 Predicted Gold Price : ₹ {prediction[0]:.2f}"

    return render_template(
        "index.html",
        min_year=MIN_YEAR,
        prediction_text=result
    )

@app.route("/chart-data")
def chart_data():

    yearly = gold.groupby("year")["Price"].mean().reset_index()

    return jsonify({
        "labels": yearly["year"].astype(str).tolist(),
        "values": yearly["Price"].tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)