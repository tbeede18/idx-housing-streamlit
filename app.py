from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import joblib
except ImportError:  # pragma: no cover - handled in the UI
    joblib = None


st.set_page_config(
    page_title="House Price Predictor",
    page_icon="H",
    layout="centered",
)


MODEL_PATH = Path("models/house_price_model.pkl")


@st.cache_resource
def load_model(model_path: Path):
    """Load a serialized model when one is available."""
    if joblib is None:
        return None, "joblib is not installed yet."

    if not model_path.exists():
        return None, f"Model file not found at: {model_path}"

    try:
        model = joblib.load(model_path)
        return model, None
    except Exception as exc:  # pragma: no cover - defensive UI fallback
        return None, f"Could not load the model: {exc}"


def predict_price(model, features: pd.DataFrame) -> float:
    """Run prediction on a single-row DataFrame."""
    prediction = model.predict(features)
    return float(prediction[0])


st.title("House Price Predictor")
st.write(
    "Enter a few home details below to generate a price prediction. "
    "The app structure is ready now, and you can connect your trained "
    "model file later."
)

with st.sidebar:
    st.header("Model Status")
    model, load_error = load_model(MODEL_PATH)

    if model is None:
        st.warning("No trained model is connected yet.")
        st.caption(load_error or "Add a saved model file to enable predictions.")
    else:
        st.success("Trained model loaded successfully.")
        st.caption(f"Using model file: `{MODEL_PATH}`")

    st.markdown("### Expected Input Features")
    st.markdown("- `LivingArea`")
    st.markdown("- `Beds`")
    st.markdown("- `Baths`")
    st.markdown("- `LotSize`")


st.subheader("Property Details")

living_area = st.number_input(
    "Living Area (square feet)",
    min_value=200,
    max_value=20000,
    value=1800,
    step=50,
)

beds = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=20,
    value=3,
    step=1,
)

baths = st.number_input(
    "Bathrooms",
    min_value=1.0,
    max_value=20.0,
    value=2.0,
    step=0.5,
)

lot_size = st.number_input(
    "Lot Size (square feet)",
    min_value=500,
    max_value=200000,
    value=7500,
    step=100,
)

input_df = pd.DataFrame(
    [
        {
            "LivingArea": living_area,
            "Beds": beds,
            "Baths": baths,
            "LotSize": lot_size,
        }
    ]
)

st.markdown("### Current Input")
st.dataframe(input_df, use_container_width=True, hide_index=True)

if st.button("Predict Price", type="primary"):
    if model is None:
        st.info(
            "Prediction is disabled until you save your trained model and place it at "
            f"`{MODEL_PATH}`."
        )
    else:
        try:
            predicted_price = predict_price(model, input_df)
            st.success(f"Estimated house price: ${predicted_price:,.2f}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


with st.expander("How to connect your trained model later"):
    st.markdown(
        """
        1. Train your model in your Jupyter notebook.
        2. Save it with `joblib.dump(model, "models/house_price_model.pkl")`.
        3. Make sure the model expects these columns in this exact order:
           `LivingArea`, `Beds`, `Baths`, `LotSize`.
        4. Run the app with `streamlit run app.py`.
        """
    )
