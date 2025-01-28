import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

transactions = pd.merge(transactions,products, on="ProductID")
customer_data = transactions.groupby("CustomerID").agg(
    total_spending=("TotalValue", "sum"),
    avg_transaction_value=("TotalValue", "mean"),
    num_transactions=("TransactionID", "count"),
    favorite_category=("Category", lambda x: x.value_counts().idxmax())
).reset_index()
customer_profiles = customers.merge(customer_data, on="CustomerID", how="left")

customer_profiles.fillna(0, inplace=True)

customer_profiles = pd.get_dummies(customer_profiles, columns=["Region", "favorite_category"])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_profiles.drop(columns=["CustomerID", "CustomerName", "SignupDate"]))
similarity_matrix = cosine_similarity(scaled_features)

# Finding top 3 similar customers for each of the first 20 customers
lookalikes = {}
for idx, customer_id in enumerate(customer_profiles["CustomerID"][:20]):
    similarities = list(enumerate(similarity_matrix[idx]))
    similarities = sorted(similarities, key=lambda x: -x[1] if x[0] != idx else -float("inf"))
    top_3 = [(customer_profiles["CustomerID"].iloc[i], score) for i, score in similarities[:3]]
    lookalikes[customer_id] = top_3

lookalike_df = pd.DataFrame([
    {"cust_id": cust_id, "lookalikes": lookalikes}
    for cust_id, lookalikes in lookalikes.items()
])
lookalike_df.to_csv("Lookalike.csv", index=False)

print("Task Successful!!")
