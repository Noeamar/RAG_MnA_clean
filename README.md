# **RAG M&A: Retrieval-Augmented Generation for Mergers and Acquisitions Analysis**

## **Description**
RAG M&A est un projet conçu pour fournir un système avancé de récupération d'informations et de génération de contenu augmentée pour analyser les données liées aux fusions et acquisitions. Ce projet combine des techniques de traitement du langage naturel (NLP) avec des bases de données d'informations commerciales, permettant de générer des insights pertinents à partir de grands volumes de données.

Le projet utilise des modèles de langage avancés (comme GPT-4) et des frameworks tels que LangChain pour :
- Extraire des informations pertinentes des fichiers CSV, PDF, ou sources en ligne.
- Générer des screenings.
- Générer des fiches d'entreprises intelligentes et contextualisés.
- Générer des fiches sur le contexte M&A d'un secteur.

---

## **Fonctionnalités**
1. **Ingestion et Préparation des Données :**
   - Lecture et nettoyage de fichiers CSV contenant des données sur les entreprises et les événements M&A.
   - Support pour les fichiers PDF, textes, et CSV via des loaders personnalisés.

2. **Splitting et Indexation des Données :**
   - Division des documents en chunks pour un traitement efficace.
   - Création d’un espace vectoriel pour indexer les embeddings des textes.

3. **Génération et Recherche Augmentée :**
   - Utilisation de modèles de langage pour répondre à des requêtes spécifiques concernant les M&A.
   - Recherche rapide d’informations pertinentes grâce à un système de vecteurs embarqué (ChromaDB).

4. **Interaction avec l’Utilisateur :**
   - Fournir des réponses précises et générées dynamiquement à des questions concernant les entreprises, les secteurs et les métriques financières.

---

## **Déploiement sur Vercel (app complète)**
Cette version remplace Streamlit par :
- une API Python (FastAPI) dans `api/index.py`
- une interface web statique (`index.html`, `app.js`, `styles.css`)

### **Pré-requis secrets Vercel**
Définir ces variables dans Vercel (Project → Settings → Environment Variables) :
- `OPENAI_API_KEY`
- `GCP_SERVICE_ACCOUNT_JSON` (JSON du service account, en une seule ligne)
- `GDRIVE_USERS_FILE_ID` (ID du fichier CSV/Google Sheet pour les inscriptions)
- optionnel : `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_ENDPOINT`

### **Déploiement**
1. Importer le repo GitHub dans Vercel.
2. Vercel détecte automatiquement `api/index.py` pour l'API et `index.html` pour le front.
3. Ouvrir l'URL Vercel et tester via l'onglet "Accueil" (statut API).

---

## **Test local via `npm run dev`**
1. Créez un fichier `.env.local` à la racine avec vos secrets :
   - `OPENAI_API_KEY`
   - `GCP_SERVICE_ACCOUNT_JSON` (JSON sur une seule ligne)
   - `GDRIVE_USERS_FILE_ID`
2. Installez les dépendances :
   - `npm install`
3. Lancez :
   - `npm run dev`

Le front tourne sur `http://localhost:3000` et l'API sur `http://localhost:8000` (proxy automatique).