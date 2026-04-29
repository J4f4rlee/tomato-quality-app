import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Tomato Quality Classifier", page_icon="🍅")

st.title("🍅 Tomato Quality Classification System")
st.write("Enter transport and storage conditions to predict tomato market status.")

# -----------------------------
# 1. Demo dataset
# -----------------------------
data = {
    "distance_km": [
        100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,
        154,154,154,154,154,154,154,
        154,154,154,154,154,154,154,
        205,205,205,205,205,205,205,
        205,205,205,205,205,205,205
    ],
    "temperature_c": [
        10,10,10,10,10,10,10,
        22,22,22,22,22,22,22,
        10,10,10,10,10,10,10,
        22,22,22,22,22,22,22,
        10,10,10,10,10,10,10,
        22,22,22,22,22,22,22
    ],
    "storage_day": [
        0,2,4,6,8,10,12,
        0,2,4,6,8,10,12,
        0,2,4,6,8,10,12,
        0,2,4,6,8,10,12,
        0,2,4,6,8,10,12,
        0,2,4,6,8,10,12
    ],
    "vibration_level": [
        0.9,0.9,0.9,0.9,0.9,0.9,0.9,
        0.9,0.9,0.9,0.9,0.9,0.9,0.9,
        1.1,1.1,1.1,1.1,1.1,1.1,1.1,
        1.1,1.1,1.1,1.1,1.1,1.1,1.1,
        1.3,1.3,1.3,1.3,1.3,1.3,1.3,
        1.3,1.3,1.3,1.3,1.3,1.3,1.3
    ],
    "weight_loss": [
        0.0,0.5,0.9,1.5,2.2,2.7,3.09,
        0.0,1.1,2.3,3.6,4.8,5.4,5.96,
        0.0,0.6,1.0,1.6,2.3,2.9,3.30,
        0.0,1.5,2.9,4.1,5.1,5.8,6.31,
        0.0,0.7,1.1,1.7,2.4,3.0,3.50,
        0.0,1.6,3.0,4.2,5.1,6.0,6.91
    ],
    "firmness_loss": [
        0.0,6.0,11.0,16.0,21.0,25.0,28.36,
        0.0,11.0,21.0,31.0,40.0,46.0,50.82,
        0.0,7.0,13.0,19.0,25.0,30.0,33.69,
        0.0,12.0,22.0,32.0,41.0,47.0,51.44,
        0.0,8.0,14.0,21.0,27.0,32.0,37.12,
        0.0,13.0,24.0,35.0,44.0,51.0,58.39
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Labeling rules
# -----------------------------
def assign_class(row):
    if row["weight_loss"] < 3 and row["firmness_loss"] < 30:
        return "Marketable"
    elif row["weight_loss"] > 5.5 or row["firmness_loss"] > 50:
        return "Unmarketable"
    else:
        return "Fast sale required"

df["quality_class"] = df.apply(assign_class, axis=1)

# -----------------------------
# 3. Train classifier
# -----------------------------
X = df[["distance_km", "temperature_c", "storage_day", "vibration_level"]]
y = df["quality_class"]

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# -----------------------------
# 4. User inputs
# -----------------------------
st.subheader("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    distance_km = st.number_input("Transport distance (km)", min_value=0, value=205)
    temperature_c = st.number_input("Storage temperature (°C)", min_value=0, value=22)

with col2:
    storage_day = st.number_input("Storage day", min_value=0, value=12)
    vibration_level = st.number_input("Vibration level", min_value=0.0, value=1.3, step=0.1)

# -----------------------------
# 5. Predict
# -----------------------------
if st.button("Predict Quality Class"):
    input_df = pd.DataFrame([{
        "distance_km": distance_km,
        "temperature_c": temperature_c,
        "storage_day": storage_day,
        "vibration_level": vibration_level
    }])

    prediction = clf.predict(input_df)[0]
    probabilities = clf.predict_proba(input_df)[0]
    classes = clf.classes_

    st.subheader("Prediction Result")

    if prediction == "Marketable":
        st.success(f"Predicted class: {prediction}")
    elif prediction == "Fast sale required":
        st.warning(f"Predicted class: {prediction}")
    else:
        st.error(f"Predicted class: {prediction}")

    st.subheader("Class Probabilities")
    prob_df = pd.DataFrame({
        "Class": classes,
        "Probability": probabilities
    }).sort_values(by="Probability", ascending=False)

    st.dataframe(prob_df, use_container_width=True)

    st.subheader("Interpretation")
    if prediction == "Marketable":
        st.write("Tomatoes are suitable for normal sale conditions.")
    elif prediction == "Fast sale required":
        st.write("Tomatoes are still saleable, but they should be sold quickly.")
    else:
        st.write("Tomatoes are predicted to be unsuitable for market sale.")

# -----------------------------
# 6. Optional dataset view
# -----------------------------
with st.expander("Show training dataset"):
    st.dataframe(df, use_container_width=True)