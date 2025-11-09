import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# Load updated dataset with recalculated scores
@st.cache_data
def load_data():
    data = pd.read_csv("Students_Grading_Dataset.csv")
    data.columns = data.columns.str.strip()
    return data.dropna()

data = load_data()

# Define features and target
target_col = 'Total_Score_Recalculated'
X = data.drop(columns=['Total_Score', target_col, 'Student_ID', 'First_Name', 'Last_Name', 'Email'])
y = data[target_col]

# Define categorical and numeric features
categorical_cols = ['Gender', 'Department', 'Extracurricular_Activities',
                    'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level', 'Grade']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Build preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ]
)

# Create full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Split data (can also do this outside app for efficiency)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (you might want to train this once and save to joblib for production)
pipeline.fit(X_train, y_train)

# Streamlit UI
st.title("🎓 Student Performance Predictor")

with st.form("input_form"):
    inputs = {}
    for col in numeric_cols:
        inputs[col] = st.number_input(col, float(data[col].min()), float(data[col].max()), float(data[col].mean()))
    for col in categorical_cols:
        options = sorted(data[col].dropna().unique())
        inputs[col] = st.selectbox(col, options)
    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([inputs])
    prediction = pipeline.predict(input_df)[0]
    st.success(f"📊 Predicted Total Score: **{prediction:.2f}**")



