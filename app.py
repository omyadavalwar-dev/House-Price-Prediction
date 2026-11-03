
import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide"
)

# ---------- CUSTOM DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: #f3f6fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1150px;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.hero {
    background: linear-gradient(120deg, #101d38, #234b83);
    padding: 38px;
    border-radius: 20px;
    color: white;
    margin-bottom: 28px;
}

.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    color: #d8e6ff;
    font-size: 16px;
}

.tag {
    color: #a9ceff;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 2px;
}

.info-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e1e8f2;
    margin-bottom: 15px;
}

.info-card h3 {
    color: #142746;
    margin: 0;
    font-size: 23px;
}

.info-card p {
    color: #718096;
    margin: 5px 0 0 0;
    font-size: 13px;
}

.section-title {
    color: #142746;
    font-size: 22px;
    font-weight: 700;
    margin: 20px 0 5px 0;
}

.stTextInput label, .stNumberInput label,
.stSelectbox label {
    color: #253858 !important;
    font-weight: 600 !important;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: #2463eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 700;
    min-height: 48px;
}

div.stFormSubmitButton > button:hover {
    background: #1749b5;
    color: white;
}

.result-card {
    background: linear-gradient(120deg, #10213e, #24538d);
    color: white;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-top: 20px;
}

.result-card p {
    color: #d5e4ff;
    font-size: 15px;
}

.result-card h1 {
    color: white;
    font-size: 40px;
}

div[data-testid="stForm"] {
    background: white;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #e1e8f2;
}
</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <div class="tag">ARTIFICIAL INTELLIGENCE • REAL ESTATE</div>
    <h1>🏡 House Price Prediction</h1>
    <p>Smart House Price Prediction powered by Machine Learning.</p>
</div>
""", unsafe_allow_html=True)


# ---------- MODEL ----------
@st.cache_resource
def train_model():

    df = pd.read_csv("HousePricePrediction.csv")

    data = df.dropna(subset=["SalePrice"]).copy()
    data = data.drop(columns=["Id"])

    data["HouseAge"] = 2010 - data["YearBuilt"]
    data["RemodAge"] = 2010 - data["YearRemodAdd"]
    data["IsRemodeled"] = (
        data["YearBuilt"] != data["YearRemodAdd"]
    ).astype(int)

    numeric_cols = [
        "MSSubClass", "LotArea", "OverallCond",
        "YearBuilt", "YearRemodAdd", "TotalBsmtSF",
        "HouseAge", "RemodAge", "BsmtFinSF2"
    ]

    categorical_cols = [
        "MSZoning", "LotConfig", "BldgType", "Exterior1st"
    ]

    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore"
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
        ("other", "passthrough", ["IsRemodeled"])
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X, y)

    return model, data, numeric_cols, categorical_cols


model, training_data, numeric_cols, categorical_cols = train_model()


# ---------- DASHBOARD CARDS ----------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <h3>1,460</h3>
        <p>Training Records</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h3>Random Forest</h3>
        <p>Machine Learning Model</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <h3>AI Powered</h3>
        <p>Price Estimation</p>
    </div>
    """, unsafe_allow_html=True)


# ---------- INPUT FORM ----------
st.markdown(
    '<div class="section-title">Property Details</div>',
    unsafe_allow_html=True
)

st.caption("Enter the property information below to estimate its value.")

with st.form("prediction_form"):

    user_input = {}

    st.markdown("**📐 Property Measurements**")

    numeric_columns = st.columns(2)

    for i, col in enumerate(numeric_cols):
        with numeric_columns[i % 2]:
            default_value = float(training_data[col].median())

            user_input[col] = st.number_input(
                col,
                value=default_value,
                step=1.0
            )

    st.divider()
    st.markdown("**🏘️ Property Characteristics**")

    categorical_columns = st.columns(2)

    for i, col in enumerate(categorical_cols):

        options = sorted(
            training_data[col].dropna().astype(str).unique()
        )

        with categorical_columns[i % 2]:
            user_input[col] = st.selectbox(
                col,
                options
            )

    st.markdown("")
    submitted = st.form_submit_button(
        "✨ Predict House Price",
        use_container_width=True
    )


# ---------- PREDICTION ----------
if submitted:

    input_df = pd.DataFrame([user_input])

    input_df["HouseAge"] = 2010 - input_df["YearBuilt"]
    input_df["RemodAge"] = 2010 - input_df["YearRemodAdd"]

    input_df["IsRemodeled"] = (
        input_df["YearBuilt"] != input_df["YearRemodAdd"]
    ).astype(int)

    prediction = model.predict(input_df)[0]

    st.markdown(f"""
    <div class="result-card">
        <p>YOUR ESTIMATED PROPERTY VALUE</p>
        <h1>${prediction:,.2f}</h1>
        <p>Predicted using Random Forest Regression</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Note: This is an estimated value based on the model, "
        "not a guaranteed market sale price."
    )


# ---------- FOOTER ----------
st.divider()

st.markdown(
    "<center style='color:#718096;'>"
    "HomeValue AI | Machine Learning Project"
    "<br>Built with Python, Scikit-learn & Streamlit"
    "</center>",
    unsafe_allow_html=True
)
