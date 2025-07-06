# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

# Page configuration
st.set_page_config(page_title="Titanic Logistic Regression App", layout="centered")

# Title
st.title("Titanic Survival Prediction App")

# -----------------------------------------------
# Step 1: Load and preprocess data
# -----------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Titanic_train.csv")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    return df

df = load_data()

# -----------------------------------------------
# Step 1c: Data Visualization
# -----------------------------------------------
st.subheader("Data Visualization")

tab1, tab2, tab3 = st.tabs([
    "Survival by Sex",
    "Fare Distribution",
    "Age vs Survival"
])

with tab1:
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Survived", hue="Sex", ax=ax1)
    ax1.set_title("Survival Count by Sex")
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df, x="Fare", bins=30, kde=True, ax=ax2)
    ax2.set_title("Fare Distribution")
    st.pyplot(fig2)

with tab3:
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x="Survived", y="Age", ax=ax3)
    ax3.set_title("Age Distribution by Survival")
    st.pyplot(fig3)

# Additional EDA
st.subheader("Additional Visualizations")

# Survival by Class
fig4, ax4 = plt.subplots()
sns.countplot(data=df, x="Pclass", hue="Survived", ax=ax4)
ax4.set_title("Survival by Passenger Class")
st.pyplot(fig4)

# Correlation Heatmap
fig5, ax5 = plt.subplots(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax5)
ax5.set_title("Feature Correlation Heatmap")
st.pyplot(fig5)

# -----------------------------------------------
# Step 2: Split Features and Labels
# -----------------------------------------------
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------
# Step 3: Train Logistic Regression Model
# -----------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------------------------
# Step 4: Model Evaluation
# -----------------------------------------------
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

st.subheader("Model Performance")
st.write("Accuracy:", accuracy_score(y_val, y_pred))
st.write("F1 Score:", f1_score(y_val, y_pred))
st.write("ROC AUC Score:", roc_auc_score(y_val, y_prob))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, y_prob)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, color='blue', label='ROC curve')
ax_roc.plot([0, 1], [0, 1], color='red', linestyle='--')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve')
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

# Probability Threshold Slider
threshold = st.slider("Adjust Classification Threshold", 0.0, 1.0, 0.5)
custom_pred = (y_prob >= threshold).astype(int)
st.write("Custom Threshold Accuracy:", accuracy_score(y_val, custom_pred))
st.write("Custom Threshold F1 Score:", f1_score(y_val, custom_pred))

# -----------------------------------------------
# Step 5: Manual Prediction
# -----------------------------------------------
st.subheader("Manual Prediction")

pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Encode input
input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": 0 if sex == "male" else 1,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked": {"S": 0, "C": 1, "Q": 2}[embarked]
}])

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    st.write(f"Survival Probability: {prob:.2f}")
    if pred == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Did Not Survive")

# -----------------------------------------------
# Step 6: Batch Prediction on Titanic_test.csv
# -----------------------------------------------
st.subheader("Batch Prediction on Titanic_test.csv")

if st.checkbox("Run model on test dataset"):
    try:
        test_df = pd.read_csv("Titanic_test.csv")

        # Preprocess
        test_df["Age"] = test_df["Age"].fillna(df["Age"].median())
        test_df["Fare"] = test_df["Fare"].fillna(df["Fare"].median())
        test_df["Embarked"] = test_df["Embarked"].fillna("S")
        test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})
        test_df["Embarked"] = test_df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

        X_test = test_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
        test_df["Predicted_Survived"] = model.predict(X_test)
        test_df["Survival_Probability"] = model.predict_proba(X_test)[:, 1]
        test_df["Sex_display"] = test_df["Sex"].map({0: "male", 1: "female"})

        display_cols = [
            "PassengerId",
            "Pclass",
            "Sex_display",
            "Age",
            "Fare",
            "Predicted_Survived",
            "Survival_Probability"
        ]

        st.write(test_df[display_cols].head())

        csv = test_df[display_cols].to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="titanic_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing test dataset: {e}")
