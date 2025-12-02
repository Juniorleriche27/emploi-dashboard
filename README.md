# Kaggle EDA Pro — Analyse exploratoire robuste

Application Streamlit pour l'analyse exploratoire de données CSV avec Assistant IA Cohere.

## Fonctionnalités

- Chargement de fichiers CSV avec gestion robuste des erreurs
- Analyse exploratoire complète (aperçu, qualité, corrélations, séries temporelles, catégories)
- Export de CSV nettoyé
- Assistant IA (Cohere) pour requêtes en langage naturel

## Lancer en local

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Déploiement sur Streamlit Cloud

1. Connectez votre dépôt GitHub à [Streamlit Cloud](https://streamlit.io/cloud)
2. Sélectionnez le dépôt `emploi-dashboard-git`
3. Fichier principal : `streamlit_app.py`
4. Ajoutez votre clé API Cohere dans les Secrets :
   - `COHERE_API_KEY = "sk_..."`
5. L'application se déploiera automatiquement !

## Configuration

- Modèles Cohere disponibles : `command-r-08-2024`, `command-r-plus-08-2024`, `command-a-03-2025`
