import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import seaborn as sns

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Purchase Predictor",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Laptop Purchase Predictor")
st.markdown("Upload your dataset, train a **Random Forest** model, and predict whether a person will buy a laptop.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Settings")
n_estimators = st.sidebar.slider("Number of Trees", 2, 50, 5)
test_size     = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100
random_state  = st.sidebar.number_input("Random State", value=42, step=1)

# ── File Upload ───────────────────────────────────────────────────────────────
st.header("📂 Upload Dataset")
uploaded_file = st.file_uploader("Upload your CSV file (must have Age, Salary, Buy_Laptop columns)", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df = df.dropna(axis=1, how='all')
    df = df.dropna()
    return df

if uploaded_file:
    data = load_data(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(data, use_container_width=True)
    st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")

    # ── Validate columns ─────────────────────────────────────────────────────
    required_cols = {'Age', 'Salary', 'Buy_Laptop'}
    if not required_cols.issubset(data.columns):
        st.error(f"❌ Dataset must contain columns: {required_cols}. Found: {list(data.columns)}")
        st.stop()

    X = data[['Age', 'Salary']]
    y = data['Buy_Laptop']

    # ── Train / Test Split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ── Train Model ──────────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    st.success(f"✅ Model trained successfully!  |  **Accuracy: {acc*100:.2f}%**")

    # ── Metrics ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Confusion Matrix")
        cm  = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['NO', 'YES'], yticklabels=['NO', 'YES'], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred, target_names=['NO', 'YES'], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(2), use_container_width=True)

    # ── Decision Trees ───────────────────────────────────────────────────────
    st.subheader("🌳 Decision Trees in the Forest")
    trees_to_show = min(n_estimators, 4)
    cols = st.columns(trees_to_show)
    for i in range(trees_to_show):
        with cols[i]:
            fig, ax = plt.subplots(figsize=(5, 8))
            tree.plot_tree(
                rf.estimators_[i],
                filled=True,
                feature_names=['Age', 'Salary'],
                class_names=['NO', 'YES'],
                ax=ax
            )
            ax.set_title(f"Tree {i+1}")
            st.pyplot(fig)
            plt.close()

    # ── Feature Importance ───────────────────────────────────────────────────
    st.subheader("🔍 Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': ['Age', 'Salary'],
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color=['#4C72B0', '#DD8452'])
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")
    st.pyplot(fig)
    plt.close()

    # ── Prediction ───────────────────────────────────────────────────────────
    st.header("🔮 Make a Prediction")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        age    = st.number_input("Enter Age",    min_value=18, max_value=80, value=32)
    with p_col2:
        salary = st.number_input("Enter Salary", min_value=10000, max_value=500000, value=55000, step=1000)

    if st.button("Predict 🚀"):
        prediction = rf.predict([[age, salary]])[0]
        proba      = rf.predict_proba([[age, salary]])[0]
        if prediction == 1:
            st.success(f"✅ **Final Output: YES** — This person is likely to BUY a laptop.  "
                       f"(Confidence: {proba[1]*100:.1f}%)")
        else:
            st.error(f"❌ **Final Output: NO** — This person is unlikely to buy a laptop.  "
                     f"(Confidence: {proba[0]*100:.1f}%)")

    # ── Download Model ───────────────────────────────────────────────────────
    st.header("💾 Download Trained Model")
    model_bytes = io.BytesIO()
    pickle.dump(rf, model_bytes)
    model_bytes.seek(0)
    st.download_button(
        label="⬇️ Download Random_Forest.pkl",
        data=model_bytes,
        file_name="Random_Forest.pkl",
        mime="application/octet-stream"
    )

else:
    st.info("👆 Please upload a CSV file to get started.")
    st.markdown("""
    **Expected CSV format:**
    | Age | Salary | Buy_Laptop |
    |-----|--------|------------|
    | 25  | 40000  | 0          |
    | 35  | 75000  | 1          |
    """)
