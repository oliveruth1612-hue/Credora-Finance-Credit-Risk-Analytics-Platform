import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Credora Finance Credit Risk Analytics Platform",
    layout="wide"
)
st.markdown("""
# 🏦 Credora Finance Credit Risk Analytics Platform
### Machine Learning-Based Loan Approval & Credit Risk Dashboard
---
""")

# Load datasets
customers = pd.read_csv("dataset/customers (1).csv")
credit = pd.read_csv("dataset/credit_history.csv")
loans = pd.read_csv("dataset/loan_applications.csv")

# Merge
merged = pd.merge(customers, credit, on="customer_id", how="left")
merged = pd.merge(merged, loans, on="customer_id", how="left")
# ---------------- Sidebar Filters ----------------

st.sidebar.title("📊 Dashboard Controls")

st.sidebar.markdown("---")

employment_options = ["All"] + sorted(merged["employment_type"].dropna().unique().tolist())

selected_employment = st.sidebar.selectbox(
    "Select Employment Type",
    employment_options
)

if selected_employment != "All":
    merged = merged[merged["employment_type"] == selected_employment]
    # ---------------- Customer Search ----------------

customer_search = st.sidebar.text_input(
    "Search Customer ID"
)

if customer_search:
    merged = merged[
        merged["customer_id"].astype(str).str.contains(
            customer_search,
            case=False,
            na=False
        )
    ]

# ---------------- KPIs ----------------

st.subheader("Project Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Customers", merged["customer_id"].nunique())

col2.metric(
    "Loan Applications",
    merged["application_id"].notna().sum()
)

col3.metric(
    "Average Credit Score",
    int(merged["credit_score"].mean())
)

col4.metric(
    "Average Income",
    f"₹{int(merged['annual_income'].mean()):,}"
)
st.divider()

# ---------------- Row 1 ----------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Loan Approval Status")

    approval = merged["approval_status"].value_counts()

    fig, ax = plt.subplots(figsize=(5,4))
    approval.plot(kind="bar", ax=ax)

    ax.set_xlabel("Status")
    ax.set_ylabel("Count")

    st.pyplot(fig)

with col2:
    st.subheader("Credit Score Distribution")

    fig2, ax2 = plt.subplots(figsize=(5,4))

    ax2.hist(
        merged["credit_score"],
        bins=20
    )

    ax2.set_xlabel("Credit Score")
    ax2.set_ylabel("Customers")

    st.pyplot(fig2)

# ---------------- Row 2 ----------------

col3, col4 = st.columns(2)

with col3:
    st.subheader("Employment Type")

    employment = merged["employment_type"].value_counts()

    fig3, ax3 = plt.subplots(figsize=(5,4))
    employment.plot(kind="bar", ax=ax3)

    st.pyplot(fig3)

with col4:
    st.subheader("Income Distribution")

    fig4, ax4 = plt.subplots(figsize=(5,4))

    ax4.hist(merged["annual_income"], bins=20)

    ax4.set_title("Income Distribution")
    ax4.set_xlabel("Annual Income")
    ax4.set_ylabel("Number of Customers")

    st.pyplot(fig4)

# ---------------- Row 3 ----------------

# Calculate Risk Score
risk_score = (
    merged["num_defaults_prior"] * 10 +
    merged["num_late_payments_90d"] * 5 +
    merged["bankruptcies"] * 20 +
    merged["credit_utilization_ratio"] * 10
)

merged["Risk Score"] = risk_score

merged["Risk Category"] = pd.cut(
    merged["Risk Score"],
    bins=[-1, 20, 50, 100],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

col5, col6 = st.columns(2)

with col5:
    st.subheader("Requested Loan Amount Distribution")

    fig5, ax5 = plt.subplots(figsize=(5,4))

    ax5.hist(merged["requested_amount"].dropna(), bins=20)

    ax5.set_title("Requested Loan Amount")
    ax5.set_xlabel("Requested Amount")
    ax5.set_ylabel("Applications")

    st.pyplot(fig5)

with col6:
    st.subheader("Risk Category Distribution")

    risk_counts = merged["Risk Category"].value_counts()

    fig6, ax6 = plt.subplots(figsize=(5,4))

    risk_counts.plot(kind="bar", ax=ax6)

    ax6.set_title("Customer Risk Categories")
    ax6.set_xlabel("Risk Category")
    ax6.set_ylabel("Customers")

    st.pyplot(fig6)
# ---------------- Loan Approval Pie Chart ----------------

st.subheader("Loan Approval Percentage")

approval_counts = merged["approval_status"].value_counts()

fig7, ax7 = plt.subplots(figsize=(6,6))

ax7.pie(
    approval_counts,
    labels=approval_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

ax7.set_title("Loan Approval Percentage")

st.pyplot(fig7)
# ---------------- Customer Data ----------------

st.subheader("📋 Customer Records")
st.write(f"Showing **{len(merged)}** records")
st.dataframe(
    merged[[
        "customer_id",
        "age",
        "gender",
        "employment_type",
        "annual_income",
        "credit_score",
        "approval_status"
    ]]
)
st.markdown("---")
st.caption(
    "© 2026 Credora Finance Credit Risk Analytics Platform | "
    "Developed by Olive Ruth | B.Tech CSE (AI)"
)
