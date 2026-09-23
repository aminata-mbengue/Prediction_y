import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np

st.set_page_config(page_title="Prédire la souscription à un dépôt à terme", layout="centered")

# ------------------------------------------------------------------
# Chargement des artefacts (mis en cache pour ne pas recharger à chaque interaction)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    encoders = jb.load('encoders.joblib')
    uniques = jb.load('uniques.joblib')
    scaler = jb.load('scaler.joblib')
    gb = jb.load('gb_model.joblib')
    return encoders, uniques, scaler, gb

encoders, uniques, scaler, gb = load_artifacts()
clasnames = uniques[9]

# ------------------------------------------------------------------
# Fonction de prédiction simple
# ------------------------------------------------------------------
def Pred_func(age, job, marital, education, housing, loan, contact,
              month, day_of_week, duration, campaign, pdays, previous, poutcome):

    job_e = encoders[0].transform([job])[0]
    marital_e = encoders[1].transform([marital])[0]
    education_e = encoders[2].transform([education])[0]
    housing_e = encoders[3].transform([housing])[0]
    loan_e = encoders[4].transform([loan])[0]
    contact_e = encoders[5].transform([contact])[0]
    month_e = encoders[6].transform([month])[0]
    day_of_week_e = encoders[7].transform([day_of_week])[0]
    poutcome_e = encoders[8].transform([poutcome])[0]

    x_new = np.array([age, job_e, marital_e, education_e, housing_e, loan_e,
                       contact_e, month_e, day_of_week_e, duration, campaign,
                       pdays, previous, poutcome_e])
    x_new = x_new.reshape(1, -1)
    x_new = scaler.transform(x_new)

    y_pred = gb.predict(x_new)
    return clasnames[y_pred[0]]

# ------------------------------------------------------------------
# Fonction de prédiction multiple (CSV)
# ------------------------------------------------------------------
def Pred_func_csv(df):
    predictions = []
    for row in df.iloc[:, :].values:
        y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4], row[5],
                            row[6], row[7], row[8], row[9], row[10], row[11],
                            row[12], row[13])
        predictions.append(y_pred)
    df = df.copy()
    df['y'] = predictions
    return df

# ------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------
st.title("Prédire la souscription à un dépôt à terme")
st.write(
    "Ce modèle permet de prédire si un client souscrit à un dépôt à terme ou non "
    "à partir des informations relatives aux campagnes de marketing."
)

tab1, tab2 = st.tabs(["Prédiction simple", "Prédiction multiple (CSV)"])

# ---------------- Onglet 1 : prédiction simple ----------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("age", min_value=0, max_value=120, value=30, step=1)
        job = st.selectbox("job", uniques[0])
        marital = st.selectbox("marital", uniques[1])
        education = st.selectbox("education", uniques[2])
        housing = st.selectbox("housing", uniques[3])
        loan = st.selectbox("loan", uniques[4])
        contact = st.selectbox("contact", uniques[5])

    with col2:
        month = st.selectbox("month", uniques[6])
        day_of_week = st.selectbox("day_of_week", uniques[7])
        duration = st.number_input("duration", min_value=0, value=0, step=1)
        campaign = st.number_input("campaign", min_value=0, value=1, step=1)
        pdays = st.number_input("pdays", min_value=-1, value=999, step=1)
        previous = st.number_input("previous", min_value=0, value=0, step=1)
        poutcome = st.selectbox("poutcome", uniques[8])

    if st.button("Prédire", type="primary"):
        try:
            resultat = Pred_func(age, job, marital, education, housing, loan,
                                  contact, month, day_of_week, duration,
                                  campaign, pdays, previous, poutcome)
            st.success(f"Prédiction : **{resultat}**")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ---------------- Onglet 2 : prédiction multiple ----------------
with tab2:
    st.write("Le fichier CSV doit contenir les colonnes, dans cet ordre : "
             "age, job, marital, education, housing, loan, contact, month, "
             "day_of_week, duration, campaign, pdays, previous, poutcome")

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            df_input = pd.read_csv(fichier)
            st.write("Aperçu du fichier importé :")
            st.dataframe(df_input.head())

            if st.button("Lancer la prédiction multiple"):
                df_result = Pred_func_csv(df_input)
                st.success("Prédictions terminées.")
                st.dataframe(df_result)

                csv_bytes = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger predictions.csv",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
