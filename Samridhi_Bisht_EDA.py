# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Inspect the datasets
def inspect_data(df, name):
    print(f"\n{name} Dataset Info:\n")
    print(df.info())
    print(f"\n{name} Dataset Head:\n")
    print(df.head())

inspect_data(customers, "Customers")
inspect_data(products, "Products")
inspect_data(transactions, "Transactions")

# Data Cleaning
# Check for missing values
print("\nMissing Values:\n")
print("Customers:\n", customers.isnull().sum())
print("Products:\n", products.isnull().sum())
print("Transactions:\n", transactions.isnull().sum())

# Handle missing values (example: dropping rows with missing data or imputing values)
customers.dropna(inplace=True)
products.dropna(inplace=True)
transactions.dropna(inplace=True)

# Convert relevant columns to appropriate data types (e.g., dates)
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])

# Exploratory Data Analysis (EDA)
# 1. Yearly distribution of signup date of customers
try:
    customers['SignupYear'] = customers['SignupDate'].dt.year
    signup_year_dist = customers['SignupYear'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    signup_year_dist.plot(kind='bar', color='lightblue')
    plt.title('Yearly Distribution of Customer Signup Dates')
    plt.xlabel('Year')
    plt.ylabel('Number of Signups')
    plt.show()
except AttributeError as e:
    print(f"Error processing yearly signup distribution: {e}")

# 2. Customer distribution by region
customer_region_dist = customers['Region'].value_counts()
plt.figure(figsize=(8, 6))
customer_region_dist.plot(kind='bar', color='skyblue')
plt.title('Customer Distribution by Region')
plt.xlabel('Region')
plt.ylabel('Number of Customers')
plt.show()

# 3. Product sales analysis by category
transactions['SumTotalValue'] = transactions['Quantity'] * transactions['Price']
product_sales = pd.merge(transactions,products, on='ProductID',how='inner')
product_category_sales = product_sales.groupby('Category')['SumTotalValue'].sum()
plt.figure(figsize=(8, 6))
product_category_sales.plot(kind='bar', color='lightgreen')
plt.title('Product Sales by Category')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.show()

# 4. Transaction frequency by customer
customer_transaction_freq = transactions['CustomerID'].value_counts()
plt.figure(figsize=(8, 6))
sns.histplot(customer_transaction_freq, kde=False, color='salmon', bins=20)
plt.title('Transaction Frequency by Customer')
plt.xlabel('Number of Transactions')
plt.ylabel('Frequency')
plt.show()

# 5. Revenue trends over time
transactions['MonthYear'] = transactions['TransactionDate'].dt.to_period('M')
revenue_trend = transactions.groupby('MonthYear')['TotalValue'].sum()
plt.figure(figsize=(10, 6))
revenue_trend.plot(kind='line', marker='o', color='purple')
plt.title('Revenue Trends Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Total Revenue')
plt.show()

# 6. Revenue by region
revenue_region = transactions.merge(customers, on='CustomerID').groupby('Region')['TotalValue'].sum()
plt.figure(figsize=(8, 6))
revenue_region.plot(kind='bar', color='orange')
plt.title('Revenue by Region')
plt.xlabel('Region')
plt.ylabel('Total Revenue')
plt.show()

