import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import gdown  # Google Drive Downloader
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 📌 Streamlit Page Config
st.set_page_config(page_title="Market Basket Analysis", layout="wide")

# 📌 Title
st.title("🛒 Market Basket Analysis")

# 📌 Google Drive File IDs (Use direct file IDs from Google Drive)
file_ids = {
    "aisles": "1NOXMtLstWWokjndeC5JjLHOa2qt12j4g",
    "departments": "1lmPkg7hYjQlD5MrPmQnYckHC5nrXW8fN",
    "order_products_prior": "1Enrbqx-rtVHkqB8y0btit4xov2GtWmLE",
    "order_products_train": "1oaaNlfCrbL03JHCfcFune6ZgUpSNseks",
    "orders": "1oPGejhDdQ3t9-X14XAh6O1ElCx_nXUzS",
    "products": "17LeYJdEdcrhXpWI6NnVHzvL8IQ3-eZPB"
}

# 📌 Function to Download and Load CSV from Google Drive using gdown
@st.cache_data
def load_data_from_drive(file_id, filename, nrows=None):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        gdown.download(url, filename, quiet=False)  # Download file
        return pd.read_csv(filename, nrows=nrows)  # Load only `nrows` rows
    except Exception as e:
        st.error(f"⚠️ Failed to download {filename}: {e}")
        return pd.DataFrame()  # Return empty DataFrame if download fails

# 📌 Load Datasets (Orders: Only First 300,000 Rows)
aisles = load_data_from_drive(file_ids["aisles"], "aisles.csv")
departments = load_data_from_drive(file_ids["departments"], "departments.csv")
order_products_prior = load_data_from_drive(file_ids["order_products_prior"], "order_products_prior.csv")
order_products_train = load_data_from_drive(file_ids["order_products_train"], "order_products_train.csv")
orders = load_data_from_drive(file_ids["orders"], "orders.csv", nrows=100000)  # Load first 1 lakh rows
products = load_data_from_drive(file_ids["products"], "products.csv")

# ✅ Debug: Show available columns in orders dataset
st.write("Columns in Orders Dataset:", orders.columns.tolist())

# 📌 Ensure 'days_since_prior_order' exists before using it
if "days_since_prior_order" in orders.columns:
    orders["days_since_prior_order"].fillna(0, inplace=True)
else:
    st.error("⚠️ 'days_since_prior_order' is missing from Orders dataset! Check if orders.csv loaded correctly.")

# Convert IDs to integer type (only if they exist)
for col in ["aisle_id", "department_id", "product_id", "order_id", "user_id"]:
    for df in [products, orders, order_products_prior]:
        if col in df.columns:
            df[col] = df[col].astype(int, errors='ignore')

# 📌 Display Raw Data (Optional)
if st.checkbox("Show Raw Data"):
    st.write(orders.head())

# 📌 Order Frequency Analysis
if "order_number" in orders.columns:
    st.subheader("📊 Order Frequency Distribution")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(orders["order_number"], bins=30, kde=True, ax=ax)
    plt.xlabel("Number of Orders per User")
    plt.ylabel("Frequency")
    plt.title("Distribution of Orders per User")
    st.pyplot(fig)
else:
    st.warning("⚠️ 'order_number' column is missing in Orders dataset!")

# 📌 Order Volume by Hour
if "order_hour_of_day" in orders.columns:
    st.subheader("🕒 Order Volume by Hour of Day")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.countplot(x=orders["order_hour_of_day"], palette="viridis", ax=ax)
    plt.xlabel("Hour of the Day")
    plt.ylabel("Number of Orders")
    plt.title("Orders Throughout the Day")
    st.pyplot(fig)
else:
    st.warning("⚠️ 'order_hour_of_day' column is missing!")

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

# 📌 Footer
st.markdown("🚀 **Built with Streamlit**")
