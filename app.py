import numpy as np
import time
import pickle
import streamlit as st

MODELS = {
    "Logistic Regressor":        pickle.load(open("models/logi_reg_hyp.pkl", "rb")),
    "Decision Tree Classifier":  pickle.load(open("models/dt_hyp.pkl",       "rb")),
    "Support Vector Classifier": pickle.load(open("models/svc_hyp.pkl",      "rb")),
    "Random Forest Classifier":  pickle.load(open("models/rf_hyp.pkl",       "rb"))
}


def render_sliders():
    radius_mean             = st.slider('radius_mean', 6.981000,   28.110000)
    texture_mean            = st.slider('texture_mean', 9.710000,   39.280000)
    perimeter_mean          = st.slider('perimeter_mean', 43.790000,  188.500000)
    area_mean               = st.slider('area_mean', 143.500000, 2501.000000)
    smoothness_mean         = st.slider('smoothness_mean', 0.052630, 0.163400)
    compactness_mean        = st.slider('compactness_mean', 0.019380, 0.345400)
    concavity_mean          = st.slider('concavity_mean', 0.000000,   0.426800)
    radius_worst            = st.slider('radius_worst', 7.930000, 36.040000)
    texture_worst           = st.slider('texture_worst', 12.020000, 49.540000)
    perimeter_worst         = st.slider('perimeter_worst', 50.410000, 251.200000)
    area_worst              = st.slider('area_worst', 185.200000, 4254.000000)
    smoothness_worst        = st.slider('smoothness_worst', 0.071170, 0.222600)
    compactness_worst       = st.slider('compactness_worst', 0.027290, 1.058000)
    concavity_worst         = st.slider('concavity_worst', 0.000000, 1.252000)
    concave_point_worst     = st.slider('concave_point_worst', 0.000000, 0.291000)
    symmetry_worst          = st.slider('symmetry_worst', 0.156500, 0.663800)
    fractal_dimension_worst = st.slider('fractal_dimension_worst', 0.055040, 0.207500)

    return [radius_mean, texture_mean, perimeter_mean, area_mean,
            smoothness_mean, compactness_mean, concavity_mean, radius_worst,
            texture_worst, perimeter_worst, area_worst, smoothness_worst,
            compactness_worst, concavity_worst, concave_point_worst,
            symmetry_worst, fractal_dimension_worst]


def predict(model, attrs):
    x = np.array([attrs])
    prediction = model.predict(x)[0]
    return prediction


def main():
    st.set_page_config(page_title="Breast Cancer Detection", page_icon="🦠")
    st.header("Brest Cancer Detection Application")
    st.caption('Multiple predictive Machine Learning models are trained\
               on Breast Cancer data for potentially ascertaining\
               between malignant and benign tumor.')
    st.subheader("Sample Images")
    st.image("./Data/sample.png")
    st.markdown("---")
    st.subheader("Feature Selection")

    attrs = render_sliders()

    model_selected = st.selectbox("Select Machine Learning Model",
                                  options=["Logistic Regressor", "Decision Tree Classifier", "Support Vector Classifier", "Random Forest Classifier"])
    model = MODELS[model_selected]
    predict_selection = st.button("Predict")
    if predict_selection:
        with st.spinner("Predicting...."):
            prediction = predict(model, attrs)
            if not prediction:
                st.success("Benign Tumor")
            else:
                st.error("Malignant Tumor")


if __name__ == "__main__":
    main()
