import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    "CustomerID":        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "TotalSpend":        [200, 250, 180, 220, 210, 800, 850, 780, 820, 900, 2000, 2200, 1900, 2100, 2300],
    "PurchaseFrequency": [3, 4, 2, 3, 4, 12, 14, 11, 13, 15, 30, 33, 28, 31, 35],
})

X = data[["TotalSpend", "PurchaseFrequency"]]
X_scaled = StandardScaler().fit_transform(X)

model = KMeans(n_clusters=3, n_init=10, random_state=42)
data["Cluster"] = model.fit_predict(X_scaled)

print(data)
print()
print("Average of each group:")
print(data.groupby("Cluster")[["TotalSpend", "PurchaseFrequency"]].mean())