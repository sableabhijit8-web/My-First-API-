
import requests
import streamlit as st


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="🛡️",
    layout="centered"
)

# IMPORTANT: Use a plain URL, not a Markdown hyperlink.
API_URL = "http://127.0.0.1:8000/predict"


# -----------------------------
# Custom styling
# -----------------------------
st.markdown("""
<style>
    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 25px;
        border-radius: 18px;
        background: linear-gradient(120deg, #173b72, #3777c8);
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e7efff;
        font-size: 16px;
    }

    .result-card {
        padding: 22px;
        border-radius: 15px;
        background: #ffffff;
        border: 1px solid #dce4f0;
        margin-top: 15px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🛡️ Insurance Premium Predictor</h1>
    <p>
        Enter your details to estimate your insurance premium category
        using a machine learning model.
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Input form
# -----------------------------
st.subheader("👤 Personal Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=119,
            value=30
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=1.0,
            max_value=300.0,
            value=65.0,
            step=0.5
        )

        height = st.number_input(
            "Height (meters)",
            min_value=0.5,
            max_value=2.49,
            value=1.70,
            step=0.01,
            format="%.2f"
        )

        income_lpa = st.number_input(
            "Annual Income (LPA)",
            min_value=0.1,
            max_value=10000.0,
            value=10.0,
            step=0.5
        )

    with col2:
        smoker = st.selectbox(
            "Are you a smoker?",
            options=[False, True],
            format_func=lambda value: "Yes" if value else "No"
        )

        city = st.text_input(
            "City",
            value="Mumbai",
            placeholder="e.g. Pune"
        )

        occupation = st.selectbox(
            "Occupation",
            options=[
                "private_job",
                "government_job",
                "business_owner",
                "freelancer",
                "student",
                "unemployed",
                "retired"
            ],
            format_func=lambda value: value.replace("_", " ").title()
        )

    submitted = st.form_submit_button(
        "🔍 Predict Insurance Premium"
    )


# -----------------------------
# Call FastAPI
# -----------------------------
if submitted:

    if not city.strip():
        st.error("Please enter your city.")
        st.stop()

    input_data = {
        "age": int(age),
        "weight": float(weight),
        "height": float(height),
        "income_lpa": float(income_lpa),
        "smoker": bool(smoker),
        "city": city.strip(),
        "occupation": occupation
    }

    try:
        with st.spinner("Analyzing your details..."):

            response = requests.post(
                API_URL,
                json=input_data,
                timeout=30
            )

        if response.ok:
            result = response.json()
            prediction_data = result.get("response", {})

            if "predicted_category" not in prediction_data:
                st.error("The API response does not contain a prediction.")
                st.json(result)
                st.stop()

            st.success("Prediction completed!")

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.subheader("📊 Prediction Result")

            predicted_category = prediction_data["predicted_category"]

            st.metric(
                label="Predicted Premium Category",
                value=predicted_category
            )

            confidence = prediction_data.get("confidence")

            if confidence is not None:
                st.metric(
                    label="Model Confidence",
                    value=f"{confidence * 100:.2f}%"
                )
                st.caption(
                    "Confidence is the model's predicted probability "
                    "for the selected category, when supported by the model."
                )
            else:
                st.info(
                    "This model does not provide class probabilities, "
                    "so confidence is unavailable."
                )

            user_details = prediction_data.get("user_details", {})

            if user_details:
                st.subheader("Calculated Details")

                c1, c2 = st.columns(2)

                c1.metric("BMI", user_details.get("bmi", "N/A"))
                c2.metric(
                    "City Tier",
                    user_details.get("city_tier", "N/A")
                )

                st.write(
                    "**Age Group:**",
                    user_details.get("age_group", "N/A")
                )

                st.write(
                    "**Lifestyle Risk:**",
                    user_details.get("lifestyle_risk", "N/A").title()
                )

            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("View complete API response"):
                st.json(result)

        else:
            st.error(f"API Error: HTTP {response.status_code}")

            try:
                st.json(response.json())
            except ValueError:
                st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI server. "
            "Start the backend using uvicorn main:app --reload."
        )

    except requests.exceptions.Timeout:
        st.error("The API took too long to respond. Please try again.")

    except requests.exceptions.RequestException as error:
        st.error(f"Request failed: {error}")

    except ValueError:
        st.error("The API returned an invalid JSON response.")


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Educational ML demonstration. Predictions are estimates, "
    "not official insurance quotes or medical advice."
)
