import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np

st.set_page_config(page_title="Prédiction Dépôt à Terme", page_icon="🏦")

# ------------------------------------------------------------------
# Chargement des objets sauvegardés (à mettre dans le même dossier
# que app.py, ou dans le repo GitHub) :
#   - encoders.joblib
#   - uniques.joblib
#   - scaler.joblib
#   - gb_model.joblib   (adapte le nom si le tien est différent,
#                           ex: 'gb_model.joblib')
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    encoders = jb.load("encoders.joblib")
    uniques = jb.load("uniques.joblib")
    scaler = jb.load("scaler.joblib")
    model = jb.load("gb_model.joblib")
    return encoders, uniques, scaler, model


encoders, uniques, scaler, model = load_artifacts()

# la dernière colonne catégorielle (index 9) correspond à la cible 'y'
clasnames = uniques[-1]  # ['no', 'yes']
labels_fr = {"no": "Pas de dépôt à terme", "yes": "Dépôt à terme souscrit"}

st.title("🏦 Prédiction de souscription à un dépôt à terme")
st.write(
    "Ce modèle prédit si un client va souscrire un dépôt à terme à partir "
    "de ses informations personnelles et des informations de la campagne marketing."
)


def predict(age, job, marital, education, housing, loan, contact, month, day_of_week,
            duration, campaign, pdays, previous, poutcome):
    job_e = encoders[0].transform([job])[0]
    marital_e = encoders[1].transform([marital])[0]
    education_e = encoders[2].transform([education])[0]
    housing_e = encoders[3].transform([housing])[0]
    loan_e = encoders[4].transform([loan])[0]
    contact_e = encoders[5].transform([contact])[0]
    month_e = encoders[6].transform([month])[0]
    day_of_week_e = encoders[7].transform([day_of_week])[0]
    poutcome_e = encoders[8].transform([poutcome])[0]

    # ordre d'entrainement : age, job, marital, education, housing, loan,
    # contact, month, day_of_week, duration, campaign, pdays, previous, poutcome
    x_new = np.array([[age, job_e, marital_e, education_e, housing_e, loan_e,
                        contact_e, month_e, day_of_week_e, duration, campaign,
                        pdays, previous, poutcome_e]])
    x_new = scaler.transform(x_new)
    y_pred = model.predict(x_new)[0]
    label = clasnames[y_pred]
    return labels_fr.get(label, label)


tab1, tab2 = st.tabs(["🔹 Prédiction simple", "📄 Prédiction multiple (CSV)"])

# --------------------------- Onglet 1 : prédiction simple -----------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Âge", min_value=17, max_value=100, value=35)
        job = st.selectbox("Profession (job)", uniques[0])
        marital = st.selectbox("Statut marital", uniques[1])
        education = st.selectbox("Niveau d'éducation", uniques[2])
        housing = st.selectbox("Prêt immobilier (housing)", uniques[3])
        loan = st.selectbox("Prêt personnel (loan)", uniques[4])
        contact = st.selectbox("Type de contact", uniques[5])

    with col2:
        month = st.selectbox("Mois du dernier contact", uniques[6])
        day_of_week = st.selectbox("Jour de la semaine", uniques[7])
        duration = st.number_input("Durée du dernier contact (secondes)", min_value=0, value=120)
        campaign = st.number_input("Nombre de contacts (campaign)", min_value=1, value=1)
        pdays = st.number_input("Jours depuis le dernier contact (999 = jamais)", min_value=0, value=999)
        previous = st.number_input("Nombre de contacts précédents", min_value=0, value=0)
        poutcome = st.selectbox("Résultat de la campagne précédente (poutcome)", uniques[8])

    if st.button("Prédire", type="primary"):
        try:
            result = predict(age, job, marital, education, housing, loan, contact,
                              month, day_of_week, duration, campaign, pdays, previous, poutcome)
            st.success(f"Prédiction : **{result}**")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# --------------------------- Onglet 2 : prédiction par CSV -----------------
with tab2:
    st.write(
        "Le fichier doit contenir les colonnes : age, job, marital, education, "
        "housing, loan, contact, month, day_of_week, duration, campaign, pdays, "
        "previous, poutcome (dans cet ordre)."
    )
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        df_new = pd.read_csv(uploaded_file)
        st.dataframe(df_new.head())

        if st.button("Lancer les prédictions"):
            try:
                predictions = []
                for row in df_new.iloc[:, :14].values:
                    y_pred = predict(*row)
                    predictions.append(y_pred)
                df_new["y_predit"] = predictions
                st.success("Prédictions terminées !")
                st.dataframe(df_new)

                csv_out = df_new.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Télécharger les résultats (CSV)",
                    data=csv_out,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Erreur lors des prédictions : {e}")
