
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Mugilidae Fish Classifier", page_icon="🐟", layout="wide")

st.title("🐟 Mugilidae Fish Species Classification System")
st.markdown("### Hybrid ANN-PSO + CNN Approach")

@st.cache_resource
def load_models():
    ann_pso = joblib.load('ann_pso_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return ann_pso, scaler, label_encoder

ann_pso, scaler, label_encoder = load_models()

st.sidebar.header("About")
st.sidebar.info("""
This system classifies 5 Mugilidae fish species:
- Planiliza subviridis
- Moolgarda seheli
- Osteomugil perusii
- Moolgarda tade
- Ellochelon vaigiensis
""")

tab1, tab2 = st.tabs(["📏 Measurement-Based", "📸 Image-Based (Coming Soon)"])

with tab1:
    st.header("Classify using 15 Morphometric Measurements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Meristic Features")
        nd1 = st.number_input("ND1_Total", min_value=0.0, value=4.0, step=1.0)
        nd2 = st.number_input("ND2_Total", min_value=0.0, value=6.0, step=1.0)
        np_val = st.number_input("NP", min_value=0.0, value=14.0, step=1.0)
        nc_val = st.number_input("NC", min_value=0.0, value=14.0, step=1.0)
        nv = st.number_input("NV_Total", min_value=0.0, value=6.0, step=1.0)
        na = st.number_input("NA_Total", min_value=0.0, value=10.0, step=1.0)
    
    with col2:
        st.subheader("Morphometric Features")
        sl = st.number_input("SL (mm)", min_value=0.0, value=150.0, step=10.0)
        pl = st.number_input("PL (mm)", min_value=0.0, value=30.0, step=5.0)
        bh = st.number_input("BH (mm)", min_value=0.0, value=35.0, step=5.0)
        hl = st.number_input("HL (mm)", min_value=0.0, value=35.0, step=5.0)
        
        st.subheader("Truss Features")
        head_truss = st.number_input("Head_Truss (mm)", min_value=0.0, value=80.0, step=10.0)
        ant_truss = st.number_input("Anterior_Truss (mm)", min_value=0.0, value=70.0, step=10.0)
        mid_truss = st.number_input("Mid_Truss (mm)", min_value=0.0, value=200.0, step=20.0)
        post_truss = st.number_input("Posterior_Truss (mm)", min_value=0.0, value=200.0, step=20.0)
        tail_truss = st.number_input("Tail_Truss (mm)", min_value=0.0, value=200.0, step=20.0)
    
    if st.button("Predict Species", key="predict"):
        features = np.array([[nd1, nd2, np_val, nc_val, nv, na, sl, pl, bh, hl,
                              head_truss, ant_truss, mid_truss, post_truss, tail_truss]])
        features_scaled = scaler.transform(features)
        prediction = ann_pso.predict(features_scaled)[0]
        species = label_encoder.inverse_transform([prediction])[0]
        probabilities = ann_pso.predict_proba(features_scaled)[0]
        
        st.success(f"### Predicted Species: {species}")
        confidence = max(probabilities) * 100
        st.progress(int(confidence))
        st.caption(f"Confidence: {confidence:.1f}%")
        
        prob_df = pd.DataFrame({'Species': label_encoder.classes_, 'Probability': probabilities})
        fig = px.bar(prob_df, x='Species', y='Probability', title='Prediction Probabilities')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Image-Based Classification")
    st.info("🚧 This feature is coming soon! Currently training CNN on fish images.")

st.markdown("---")
st.markdown("<div style='text-align: center;'>Hybrid ANN-PSO System | FYP Project</div>", unsafe_allow_html=True)
