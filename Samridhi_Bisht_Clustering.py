import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")

transaction_summary = transactions.groupby("CustomerID").agg(
    total_spending=("TotalValue", "sum"),
    avg_transaction_value=("TotalValue", "mean"),
    num_transactions=("TransactionID", "count")
).reset_index()

customer_profiles = customers.merge(transaction_summary, on="CustomerID", how="left")
customer_profiles.fillna(0, inplace=True)

features = customer_profiles[["total_spending", "avg_transaction_value", "num_transactions"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

ssd = [] 
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    ssd.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, ssd, marker='o', linestyle='--')
plt.title("Elbow Method to Determine Optimal K")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Distances (SSD)")
plt.xticks(k_values)
plt.show()

optimal_k = int(input("Enter the optimal number of clusters from the elbow plot: "))
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_profiles["Cluster"] = kmeans.fit_predict(scaled_features)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_features)
customer_profiles["PCA1"] = reduced_data[:, 0]
customer_profiles["PCA2"] = reduced_data[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(
    customer_profiles["PCA1"],
    customer_profiles["PCA2"],
    c=customer_profiles["Cluster"],
    cmap="viridis",
    s=100,
    alpha=0.7
)
plt.title(f"Customer Clusters (k={optimal_k})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

customer_profiles[["CustomerID", "Cluster"]].to_csv("Customer_Clusters.csv", index=False)
print(f"Clustering completed. Results saved in 'Customer_Clusters.csv'")
