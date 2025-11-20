import requests
import streamlit as st
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("iris-frontend")

# URL du backend FastAPI
BACKEND_URL = os.getenv("BACKEND_URL") #, "http://localhost:8001")
logger.info(f"Frontend started. Using backend URL: {BACKEND_URL}")

st.set_page_config(
    page_title="Iris Predictor",
    page_icon="🌸",
    layout="centered",
)

# --- Titre & description ---
st.title("🌸 Iris Predictor")
st.markdown(
    """
Ce petit outil permet de tester un **modèle de classification Iris**.

Choisis les quatre caractéristiques de la fleur puis clique sur **Prédire** pour obtenir la classe prédite.
"""
)

with st.expander("ℹ️ À propos des features"):
    st.write(
        """
    - **Sepal length** : longueur du sépale (cm)  
    - **Sepal width** : largeur du sépale (cm)  
    - **Petal length** : longueur du pétale (cm)  
    - **Petal width** : largeur du pétale (cm)  
    """
    )

# --- Valeurs par défaut & plages raisonnables pour Iris ---
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1, 0.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5, 0.1)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4, 0.1)
petal_width = st.slider("Petal width (cm)", 0.0, 2.5, 0.2, 0.1)

# Mapping id de classe -> nom lisible
CLASS_LABELS = {
    0: "Iris setosa",
    1: "Iris versicolor",
    2: "Iris virginica",
}


def call_backend(features):
    """Appelle l'API FastAPI /predict avec la liste de features."""
    logger.info(f"Sending prediction request to backend: {features}")

    try:
        resp = requests.post(
            f"{BACKEND_URL}/predict",
            json={"features": features},
            timeout=5,
        )
        logger.info(f"Backend responded with status code: {resp.status_code}")
        resp.raise_for_status()  # if status is 4xx/5xx -> exception
        logger.info(f"Response content: {resp.text}")
        return resp.json()

    except requests.exceptions.Timeout:
        logger.error("Backend request timed out")
        st.error("⏱️ Le backend prend trop de temps à répondre.")
        return None

    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to backend")
        st.error("❌ Impossible de se connecter au backend.")
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Unexpected error calling backend: {e}")
        st.error(f"Erreur lors de l'appel à l'API : {e}")
        return None


# --- Disposition en colonnes ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎛 Paramètres sélectionnés")

    selected_params = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    logger.info(f"User selected parameters: {selected_params}")
    st.write(selected_params)

with col2:
    st.markdown("### 🔮 Prédiction")
    predict_btn = st.button("Prédire", use_container_width=True)

    if predict_btn:
        features = [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width,
        ]

        logger.info(f"'Predict' button pressed. Features: {features}")

        with st.spinner("Appel du modèle en cours..."):
            result = call_backend(features)

        if result is not None and "prediction" in result:
            class_id = int(result["prediction"])
            class_name = CLASS_LABELS.get(class_id, f"Classe inconnue ({class_id})")

            logger.info(f"Prediction successful: {result}")

            st.success("Prédiction réussie !")
            st.markdown(
                f"""
                **Classe prédite :** `{class_id}`  
                **Espèce :** **{class_name}**
                """
            )

        elif result is not None:
            logger.warning(f"Unexpected backend response: {result}")
            st.error(f"Réponse inattendue du backend : {result}")

# Footer léger
st.markdown("---")
st.caption("Demo Iris ML • FastAPI + Streamlit")
