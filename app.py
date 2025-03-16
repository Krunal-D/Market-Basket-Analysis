import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 📌 Streamlit Page Config
st.set_page_config(page_title="Market Basket Analysis", layout="wide")

# 📌 Title
st.title("🛒 Market Basket Analysis")

# 📌 Google Drive File IDs
file_ids = {
    "aisles": "1NOXMtLstWWokjndeC5JjLHOa2qt12j4g",
    "departments": "1lmPkg7hYjQlD5MrPmQnYckHC5nrXW8fN",
    "order_products_prior": "1Enrbqx-rtVHkqB8y0btit4xov2GtWmLE",
    "order_products_train": "1oaaNlfCrbL03JHCfcFune6ZgUpSNseks",
    "orders": "1oPGejhDdQ3t9-X14XAh6O1ElCx_nXUzS",
    "products": "17LeYJdEdcrhXpWI6NnVHzvL8IQ3-eZPB"
}

# 📌 Function to Download CSV from Google Drive
@st.cache_data
def load_data_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(url)

# 📌 Load Datasets
aisles = load_data_from_drive(file_ids["aisles"])
departments = load_data_from_drive(file_ids["departments"])
order_products_prior = load_data_from_drive(file_ids["order_products_prior"])
order_products_train = load_data_from_drive(file_ids["order_products_train"])
orders = load_data_from_drive(file_ids["orders"])
products = load_data_from_drive(file_ids["products"])

# ✅ Debug: Print available columns in the orders dataset
st.write("Columns in Orders Dataset:", orders.columns.tolist())

# 📌 Check if 'days_since_prior_order' column exists before accessing it
if "days_since_prior_order" in orders.columns:
    orders["days_since_prior_order"].fillna(0, inplace=True)
else:
    st.error("⚠️ Column 'days_since_prior_order' is missing from Orders dataset!")

# Convert IDs to integer type
for col in ["aisle_id", "department_id", "product_id", "order_id", "user_id"]:
    for df in [products, orders, order_products_prior]:
        if col in df.columns:
            df[col] = df[col].astype(int, errors='ignore')

# 📌 Display Raw Data (Optional)
if st.checkbox("Show Raw Data"):
    st.write(orders.head())

# 📌 Order Frequency Analysis
st.subheader("📊 Order Frequency Distribution")
fig, ax = plt.subplots(figsize=(12, 5))
sns.histplot(orders["order_number"], bins=30, kde=True, ax=ax)
plt.xlabel("Number of Orders per User")
plt.ylabel("Frequency")
plt.title("Distribution of Orders per User")
st.pyplot(fig)

# 📌 Order Volume by Hour
st.subheader("🕒 Order Volume by Hour of Day")
fig, ax = plt.subplots(figsize=(12, 5))
sns.countplot(x=orders["order_hour_of_day"], palette="viridis", ax=ax)
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Orders")
plt.title("Orders Throughout the Day")
st.pyplot(fig)

# 📌 Top 10 Most Ordered Products
st.subheader("🏆 Top 10 Most Ordered Products")
top_products = order_products_prior["product_id"].value_counts().head(10)
top_products_df = products[products["product_id"].isin(top_products.index)].copy()
top_products_df["order_count"] = top_products_df["product_id"].map(top_products)

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(y=top_products_df["product_name"], x=top_products_df["order_count"], palette="magma", ax=ax)
plt.xlabel("Number of Times Ordered")
plt.ylabel("Product Name")
plt.title("Top 10 Most Ordered Products")
st.pyplot(fig)

# 📌 Market Basket Analysis
st.subheader("📈 Market Basket Analysis")

# Convert transaction data for Apriori Algorithm
transactions = order_products_prior.groupby("order_id")["product_id"].apply(list)
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Generate frequent itemsets
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

st.write("Top Association Rules:")
st.write(rules.sort_values(by="lift", ascending=False).head(10))

# 📌 Association Rule Network
st.subheader("🔗 Association Rule Network")
strong_rules = rules[rules["lift"] > 1.2]

if not strong_rules.empty:
    fig, ax = plt.subplots(figsize=(12, 6))
    G = nx.from_pandas_edgelist(strong_rules, 'antecedents', 'consequents')
    nx.draw(G, with_labels=True, node_color="skyblue", edge_color="gray", ax=ax)
    st.pyplot(fig)
else:
    st.write("No strong rules found.")

# 📌 Department-wise Orders
st.subheader("🏬 Orders by Department")
merged_products = order_products_prior.merge(products, on="product_id", how="left")
merged_products = merged_products.merge(departments, on="department_id", how="left")

fig, ax = plt.subplots(figsize=(12, 5))
sns.countplot(y=merged_products["department"], order=merged_products["department"].value_counts().index, palette="coolwarm", ax=ax)
plt.xlabel("Number of Orders")
plt.ylabel("Department")
plt.title("Top Departments by Order Volume")
st.pyplot(fig)

# 📌 Order Heatmap
st.subheader("🔥 Order Heatmap: Hour vs. Days Since Prior Order")

# ✅ Check if 'days_since_prior_order' exists before heatmap
if "days_since_prior_order" in orders.columns:
    heatmap_data = orders.pivot_table(index="order_hour_of_day", columns="days_since_prior_order", aggfunc="size", fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="Blues", linewidths=0.5, ax=ax)
    plt.xlabel("Days Since Prior Order")
    plt.ylabel("Hour of the Day")
    plt.title("Order Heatmap")
    st.pyplot(fig)
else:
    st.warning("⚠️ Skipping heatmap: 'days_since_prior_order' column not found.")

# 📌 Footer
st.markdown("🚀 **Built with Streamlit**")
