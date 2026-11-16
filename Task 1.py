import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)
n = 200
data = pd.DataFrame({
    "sqft": np.random.randint(600, 4000, n),
    "bedrooms": np.random.randint(1, 6, n),
    "bathrooms": np.random.randint(1, 4, n),
})
data["price"] = (
    50000
    + data["sqft"] * 150
    + data["bedrooms"] * 10000
    + data["bathrooms"] * 15000
    + np.random.normal(0, 20000, n)
)

X = data[["sqft", "bedrooms", "bathrooms"]]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"R² score: {r2_score(y_test, predictions):.3f}")
print(f"Mean absolute error: ${mean_absolute_error(y_test, predictions):,.0f}")

print("\nIntercept:", round(model.intercept_, 2))
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:,.2f}")

new_house = pd.DataFrame({"sqft": [2000], "bedrooms": [3], "bathrooms": [2]})
predicted_price = model.predict(new_house)[0]
print(f"\nPredicted price for a 2000 sqft, 3 bed, 2 bath house: ${predicted_price:,.0f}")