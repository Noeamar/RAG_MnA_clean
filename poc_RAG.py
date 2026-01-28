#!/usr/bin/env python
# coding: utf-8

import os
import json
from json import dumps, loads
from operator import itemgetter
import datetime
import pandas as pd
import bs4
import time
import random
import requests  # Pour télécharger depuis GitHub
from time import sleep
# Importations de LangChain et autres
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from docxtpl import DocxTemplate
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever

# --- Configuration des variables d'environnement ---
# Laissez ces variables définies via les secrets Vercel/local.
if os.getenv("LANGCHAIN_TRACING_V2"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
if os.getenv("LANGCHAIN_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# On récupère la clé OPENAI_API_KEY depuis l'environnement
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError(
        "La variable d'environnement OPENAI_API_KEY n'est pas définie. "
        "Veuillez la définir dans vos secrets."
    )
os.environ["OPENAI_API_KEY"] = openai_api_key

# --- Téléchargement depuis GitHub ---
# Définissez l'URL de base de votre dépôt public GitHub (raw)
GITHUB_BASE_URL = "https://raw.githubusercontent.com/Noeamar/RAG_MnA/main/"

def download_file_from_github(source_blob_name: str, destination_file_name: str):
    """
    Télécharge un fichier depuis GitHub via son URL brute et le sauvegarde localement.
    
    :param source_blob_name: Chemin relatif dans le repo (ex: "FAISS_index/index.faiss")
    :param destination_file_name: Chemin local où sauvegarder le fichier.
    """
    url = GITHUB_BASE_URL + source_blob_name
    print(f"[LOG] Téléchargement de {url} vers {destination_file_name}...", flush=True)
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"[LOG] Erreur lors du téléchargement : code {response.status_code}")
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    with open(destination_file_name, "wb") as f:
        f.write(response.content)
    os.chmod(destination_file_name, 0o777)
    file_size = os.stat(destination_file_name).st_size
    print(f"[LOG] Téléchargement terminé. Taille du fichier: {file_size} octets", flush=True)
    if file_size == 0:
        raise ValueError(f"[LOG] Le fichier {destination_file_name} est vide après téléchargement.")

# --- Fonctions RAG Fusion (sans lien avec des graphes) ---
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

def load_nlp_text_documents_from_github(github_raw_url: str) -> list:
    """
    Télécharge un fichier texte depuis GitHub (URL raw) où chaque section commence par "Ligne <num>:"
    et renvoie une liste de Documents.
    
    Args:
        github_raw_url (str): URL raw du fichier.
        
    Returns:
        list: Liste de Documents.
    """
    response = requests.get(github_raw_url)
    if response.status_code != 200:
        raise Exception(f"Erreur lors du téléchargement du fichier : {response.status_code}")
    content = response.text
    segments = re.split(r'\n\s*Ligne \d+:', content)
    documents = []
    for segment in segments:
        seg = segment.strip()
        if seg:
            documents.append(Document(page_content=seg, metadata={}))
    return documents

# Exemple d'import depuis GitHub
documents = load_nlp_text_documents_from_github("https://raw.githubusercontent.com/Noeamar/RAG_MnA/main/Data/deals_data_cleaned_CFNews_converted_NLP.txt")

def rag_fusion(question: str) -> str:
    print("[LOG] Démarrage de rag_fusion pour la question :", question, flush=True)
    # Définir le répertoire et le chemin du fichier local
    local_index_dir = "./Data/FAISS_index"
    local_index_file = os.path.join(local_index_dir, "index.faiss")
    github_file_path = "index.faiss"  # Chemin relatif dans votre repo GitHub
    
    # Forcer la suppression du fichier local s'il existe pour forcer le téléchargement
    if os.path.exists(local_index_file):
        os.remove(local_index_file)
        print("[LOG] Fichier existant supprimé pour forcer le téléchargement depuis GitHub.", flush=True)
    
    # Télécharger le fichier depuis GitHub
    download_file_from_github(github_file_path, local_index_file)
    
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(local_index_dir, embeddings=embedding, allow_dangerous_deserialization=True)
    print("[LOG] Index FAISS chargé.", flush=True)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "score_threshold": 0.01})
    
    query_generation_template = """You are a seasoned M&A consultant with access to a broad dataset that includes recent news, company profiles, and investment fund details. Given the user's question:

{question}

Generate exactly 4 focused queries that will help retrieve the most relevant and comprehensive information from this rich dataset.
"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)
    generate_queries = (prompt_rag_fusion
                        | ChatOpenAI(model="gpt-5-mini")
                        | StrOutputParser()
                        | (lambda x: x.split("\n")))
    queries = generate_queries.invoke({"question": question})
    print("[LOG] Requêtes générées :", queries, flush=True)
    
    results = [retriever.invoke(q) for q in queries]
    print("[LOG] Documents récupérés :", results, flush=True)
    
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
            doc_str = dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 60)
    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [loads(d_str)]
    ]
    print(f"[LOG] Documents fusionnés : {len(reranked_docs)} documents rerankés.", flush=True)
    
    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])
    
    answer_template = """You are an M&A expert who can analyze recent news, company data, and fund information to provide a comprehensive and accurate answer. Using the following context extracted from various M&A-related sources (news, companies, funds), answer the user's question concisely and factually. Highlight relevant deals, company details, or fund strategies if mentioned. Do not invent information that isn't provided. Always give your source.

Context:
{context}

Question: {question}

Provide a clear, fact-based answer focusing on the M&A domain.
"""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model="gpt-5-mini")
    final_input = {"context": context, "question": question}
    answer = (answer_prompt | llm | StrOutputParser()).invoke(final_input)
    
    print("[LOG] Réponse générée.", flush=True)
    return answer


import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain.load import dumps, loads
from langchain.retrievers import BM25Retriever  # si votre version diffère, ajustez ce import

# Monkey‐patch si nécessaire
if not hasattr(openai, "OpenAI"):
    openai.OpenAI = openai.Client
if not hasattr(openai, "AsyncOpenAI") and hasattr(openai, "AsyncClient"):
    openai.AsyncOpenAI = openai.AsyncClient

def rag_fusion_actualites(question: str) -> str:
    print("[LOG] Démarrage pour la question :", question, flush=True)

    # 1) Dossiers de vos 3 batches FAISS
    batch_dirs = [
        "./Data/FAISS_index_actualites_NLP_400_0_batch_1",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_2",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_3",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_4",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_5",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_6",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_7"
    ]

    # 2) Embeddings
    embedding = OpenAIEmbeddings()

    # 3) Chargement FAISS en parallèle
    dense_retrievers = []
    with ThreadPoolExecutor(max_workers=len(batch_dirs)) as executor:
        future_to_path = {
            executor.submit(
                FAISS.load_local,
                path,
                embeddings=embedding,
                allow_dangerous_deserialization=True
            ): path
            for path in batch_dirs
        }
        for fut in as_completed(future_to_path):
            path = future_to_path[fut]
            try:
                vs = fut.result()
                print(f"[LOG] Index FAISS chargé depuis {path}", flush=True)
                dense_retrievers.append(
                    vs.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"score_threshold": 0.6, "k": 10}
                    )
                )
            except Exception as e:
                print(f"[ERROR] Échec chargement FAISS {path}: {e}", flush=True)

    print(f"[LOG] {len(dense_retrievers)} retrievers denses prêts.", flush=True)

    # 4) BM25Retriever (documents doit être défini en amont)
    #    ex : documents = load_nlp_text_documents(...)
    bm25_retriever = BM25Retriever.from_documents(documents, k=10)
    print("[LOG] BM25Retriever initialisé.", flush=True)

    # 5) Génération des 3 requêtes
    query_tpl = ChatPromptTemplate.from_template(
        "You are a helpful assistant that generates 3 search queries based on the input. \n"
        "Generate 3 queries related to: {question}"
    )
    raw = (
        query_tpl
        | ChatOpenAI(model="gpt-5-mini", temperature=1)
        | StrOutputParser()
    ).invoke({"question": question})
    queries = [q.strip() for q in raw.split("\n") if q.strip()]
    print("[LOG] Requêtes générées :", queries, flush=True)

    # 6) Pour chaque requête : retrieval dense+BM25 en parallèle
    all_results = []
    for q in queries:
        print(f"[LOG] Traitement de la requête : {q}", flush=True)
        with ThreadPoolExecutor(max_workers=len(dense_retrievers) + 1) as executor:
            future_to_source = {}

            # dense
            for retr in dense_retrievers:
                future_to_source[executor.submit(retr.invoke, q)] = "dense"
            # sparse
            future_to_source[executor.submit(bm25_retriever.invoke, q)] = "sparse"

            results = []
            for fut in as_completed(future_to_source):
                src = future_to_source[fut]
                try:
                    docs = fut.result()
                    print(f"  [LOG] {len(docs)} docs {src}", flush=True)
                    results.extend(docs)
                except Exception as e:
                    print(f"  [ERROR] retrieval {src} échoué : {e}", flush=True)

        all_results.append(results)

    print("[LOG] Récupération terminée pour chaque requête.", flush=True)

    # 7) Fusion RRF par requête et global
    #    Ici on simplifie en fusionnant tous ensemble comme précédemment
    global_scores = {}
    unique_docs = {}
    for docs in all_results:
        # rrf par requête
        for rank, doc in enumerate(docs, start=1):
            key = doc.page_content
            global_scores[key] = global_scores.get(key, 0) + 1.0 / (rank + 100)
            unique_docs[key] = doc

    final_docs = sorted(
        unique_docs.values(),
        key=lambda d: global_scores.get(d.page_content, 0),
        reverse=True
    )
    print(f"[LOG] Documents fusionnés globalement : {len(final_docs)} items.", flush=True)

    # 8) Construire le contexte (jusqu’aux 50 meilleurs)
    context = "\n\n".join(d.page_content for d in final_docs[:50])

    # 9) Prompt final
    answer_tpl = ChatPromptTemplate.from_template(
        "You are a financial journalist and M&A expert focusing on recent news. "
        "Using the context below, answer chronologically and citez toujours la source.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    answer = (
        answer_tpl
        | ChatOpenAI(model="gpt-5-mini", temperature=1)
        | StrOutputParser()
    ).invoke({"context": context, "question": question})

    print("[LOG] Réponse générée.", flush=True)
    return answer

import openai

def rag_fusion_actualites_search_preview(question: str) -> str:
    """
    Interroge le modèle 'gpt-4o-mini-search-preview' 
    qui intègre directement la recherche sur vos sources (Arx, CFNews, etc.),
    et renvoie une synthèse chronologique des actualités.
    """
    print("[LOG] Démarrage web-search-preview pour la question :", question, flush=True)

    # 1) Template du prompt final
    answer_template = (
        "You are a financial journalist and M&A expert with direct access "
        "to the internet. Perform an internal search for all relevant "
        "news items, then answer chronologically, always citing your sources. You must find all the sources from the 5 precedent years (2020-2025) and list them as bullet points with just the title of the article source. \n\n"
        "Question: {question}"
    )

    # 2) Construction du prompt
    prompt_final = answer_template.format(question=question)
    print("[LOG] Prompt final construit :", prompt_final, flush=True)

    # 3) Appel du LLM avec capacité de recherche intégrée
    resp = openai.chat.completions.create(
        model="gpt-4o-mini-search-preview", 
        web_search_options={
            "search_context_size": "high"
        },
        messages=[{"role": "user", "content": prompt_final}],
    )
    raw = resp.choices[0].message.content
    print("[LOG] Réponse brute reçue.", flush=True)

    return raw

    
def rag_fusion_fonds(question: str) -> str:
    print("[LOG] Démarrage de rag_fusion_fonds pour la question :", question)
    local_index_dir = "./Data/FAISS_index_fonds"
    local_index_file = os.path.join(local_index_dir, "index.faiss")
    github_file_path = "FAISS_index_fonds/index.faiss"
    if not os.path.exists(local_index_file):
        download_file_from_github(github_file_path, local_index_file)
    else:
        print(f"[LOG] Fichier index fonds déjà présent : {local_index_file}")
    
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(local_index_dir, embeddings=embedding, allow_dangerous_deserialization=True)
    print("[LOG] Index fonds chargé.")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 20, "score_threshold": 0.01})
    
    query_generation_template = """You are an expert in private equity and investment funds. Given the user's query:

{question}

Generate five alternative versions of the question to retrieve relevant documents from a vector database.
Provide these alternative questions separated by newlines.
"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)
    generate_queries = (prompt_rag_fusion
                        | ChatOpenAI(model="gpt-5-mini", temperature=1)
                        | StrOutputParser()
                        | (lambda x: x.split("\n")))
    queries = generate_queries.invoke({"question": question})
    print("[LOG] Requêtes générées :", queries)
    
    results = [retriever.invoke(q) for q in queries]
    print("[LOG] Documents récupérés :", results)
    
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
            doc_str = dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 60)
    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [loads(d_str)]
    ]
    print(f"[LOG] Documents fusionnés : {len(reranked_docs)} documents rerankés.")
    
    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])
    
    answer_template = """You are a private equity and investment fund specialist. Using the following context sourced from a database of investment funds, answer the user's question focusing on fund characteristics such as ticket size, sector preferences, geographic focus, and investment strategies. Provide a factual and concise explanation based solely on the provided information. Always indicate the source (Arx).

Context:
{context}

Question: {question}

Offer a fact-based response highlighting key investment criteria.
"""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model="gpt-5-mini", temperature=1)
    final_input = {"context": context, "question": question}
    answer = (answer_prompt | llm | StrOutputParser()).invoke(final_input)
    
    print("[LOG] Réponse fonds générée.")
    return answer

def rag_fusion_fiche_societe_to_word(question: str) -> dict:
    """
    Interroge la base M&A (RAG via FAISS) et retourne une fiche JSON pour Word.
    """
    print("[LOG] Démarrage rag_fusion_fiche_societe_to_word pour :", question)

    # 1) Chemins vers vos 3 batches FAISS
    batch_dirs = [
        "./Data/FAISS_index_actualites_NLP_400_0_batch_1",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_2",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_3",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_4",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_5",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_6",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_7",
    ]

    # 2) Chargement parallèle des retrievers
    embedding = OpenAIEmbeddings()
    retrievers = []
    with ThreadPoolExecutor(max_workers=len(batch_dirs)) as exe:
        fut_to_path = {
            exe.submit(
                FAISS.load_local,
                path,
                embeddings=embedding,
                allow_dangerous_deserialization=True
            ): path
            for path in batch_dirs
        }
        for fut in as_completed(fut_to_path):
            path = fut_to_path[fut]
            try:
                vs = fut.result()
                retrievers.append(
                    vs.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"score_threshold": 0.6, "k": 10}
                    )
                )
                print(f"[LOG] Chargé FAISS depuis {path}")
            except Exception as e:
                print(f"[ERROR] échec chargement {path} : {e}")

    print(f"[LOG] {len(retrievers)} retrievers prêts.")

    # 3) Génération de 3 requêtes (modèle gpt-5-mini, température par défaut)
    prompt_q = f"""
You are a helpful assistant that generates 3 distinct search queries based on the input.
Input: {question}

Output the 3 queries, one per line:
""".strip()
    resp_q = openai.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt_q}],
    )
    raw_q = resp_q.choices[0].message.content
    queries = [q.strip() for q in raw_q.splitlines() if q.strip()]
    print("[LOG] Queries générées :", queries)

    # 4) Recherche sur chaque retriever + RRF par requête
    all_docs = []
    for q in queries:
        docs = []
        with ThreadPoolExecutor(max_workers=len(retrievers)) as exe:
            futs = [exe.submit(r.invoke, q) for r in retrievers]
            for f in as_completed(futs):
                try:
                    docs.extend(f.result())
                except Exception as e:
                    print(f"[ERROR] retrieval '{q}' failed : {e}")
        all_docs.append(docs)

    # 5) Fusion RRF globale
    fused = {}
    for docs in all_docs:
        for rank, doc in enumerate(docs, start=1):
            key = json.dumps({"page_content": doc.page_content, "metadata": doc.metadata})
            fused[key] = fused.get(key, 0.0) + 1.0 / (rank + 100)
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    reranked_docs = [Document(**json.loads(k)) for k, _ in ranked]
    print(f"[LOG] RRF a produit {len(reranked_docs)} docs.")

    # 6) Construire le contexte complet
    context = "\n\n".join(d.page_content for d in reranked_docs)

    # 7) Prompt final
    answer_template = r"""
You are a financial journalist and M&A expert. You MUST answer in JSON format only, strictly matching the provided structure.

Context:
{context}

Question: {question}

Rewrite everything you are given in the context into well-formed sentences.

Respond ONLY within the following structure (no extra text):

{{
    "nom_societe": "Provide the company name if found",
    "description_activite": "Provide a detailed description of the company's activities (5-10 lines), using clear and concise language.",
    "chiffres_cles": "Include key metrics such as revenue, employee count, or founding date, summarized if needed.",
    "clients_par_secteur": "List the main clients by sector and give their name.",
    "implantation_positionnement": "List cities or countries where the company is located",
    "elements_financiers": "Summarize the financial growth over the past 3 years in a concise manner",
    "president": "Name of the president",
    "daf": "Name of the financial director",
    "actionnaire": "Provide a summarized list of key shareholders or investment funds, with the most important ones highlighted",
    "actionnaire_pourcentage": "Shareholder distribution percentages if available",
    "creanciers_type": "List types of creditors concisely",
    "creanciers_commentaires": "Provide a brief summary of the creditors' comments",
    "actualites_presse": "Present recent press news with maximum details and clear language",
    "equity_story": "Present major equity events and investments with maximum details and clear language",
    "creation": "Present creation details or year of founding with maximum details and clear language",
    "acquisitions": "Present all key acquisitions, including dates and descriptions, with maximum details and clear language"
}}
""".strip()

    prompt_final = answer_template.format(context=context, question=question)
    resp_f = openai.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt_final}],
    )
    raw_f = resp_f.choices[0].message.content

    # 8) Nettoyer et parser
    text = raw_f.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("[ERROR] JSON parse failed:", e)
        return {}

def rag_fusion_fiche_societe_to_word_websearch(question: str) -> dict:
    """
    Interroge votre base FAISS RAG + websearch-preview pour produire une fiche JSON.
    """
    print("[LOG] Démarrage pour :", question)

    # 1) Chemins vers vos 7 batches FAISS
    batch_dirs = [
        "./Data/FAISS_index_actualites_NLP_400_0_batch_1",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_2",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_3",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_4",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_5",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_6",
        "./Data/FAISS_index_actualites_NLP_400_0_batch_7",
    ]

    # 2) Chargement parallèle des retrievers
    embedding = OpenAIEmbeddings()
    retrievers = []
    with ThreadPoolExecutor(max_workers=len(batch_dirs)) as exe:
        future_to_dir = {
            exe.submit(
                FAISS.load_local,
                path,
                embeddings=embedding,
                allow_dangerous_deserialization=True
            ): path
            for path in batch_dirs
        }
        for fut in as_completed(future_to_dir):
            path = future_to_dir[fut]
            try:
                vs = fut.result()
                retrievers.append(
                    vs.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"score_threshold": 0.6, "k": 10}
                    )
                )
                print(f"[LOG] Chargé FAISS depuis {path}")
            except Exception as e:
                print(f"[ERROR] échec chargement {path} : {e}")

    print(f"[LOG] {len(retrievers)} retrievers prêts.")

    # 3) Génération de 3 requêtes avec gpt-5-mini (température par défaut = 1)
    prompt_q = f"""
You are a helpful assistant that generates 3 distinct search queries based on the input.
Input: {question}

Output the 3 queries, one per line:
""".strip()
    resp_q = openai.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt_q}],
    )
    raw_q = resp_q.choices[0].message.content
    queries = [q.strip() for q in raw_q.splitlines() if q.strip()]
    print("[LOG] Queries générées :", queries)

    # 4) Recherche parallèle sur chaque retriever + RRF par requête
    all_docs = []
    for q in queries:
        with ThreadPoolExecutor(max_workers=len(retrievers)) as exe:
            futs = [exe.submit(r.invoke, q) for r in retrievers]
            docs = []
            for f in as_completed(futs):
                try:
                    docs.extend(f.result())
                except Exception as e:
                    print(f"[ERROR] retrieval '{q}' failed : {e}")
        all_docs.append(docs)

    # 5) RRF global
    fused = {}
    for docs in all_docs:
        for rank, doc in enumerate(docs, start=1):
            key = json.dumps({"page_content": doc.page_content, "metadata": doc.metadata})
            fused[key] = fused.get(key, 0) + 1.0 / (rank + 100)
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    reranked_docs = [Document(**json.loads(k)) for k, _ in ranked]
    print(f"[LOG] RRF fusion a produit {len(reranked_docs)} docs.")

    # 6) Contexte abrégé (10 premiers)
    context = "\n\n".join(doc.page_content for doc in reranked_docs[:10])

    # 7) Prompt final avec web-search-preview, sans temperature
    answer_template = r"""
You are a financial journalist and M&A expert. You MUST answer in JSON format only, strictly matching the provided structure.

Use primarily the context provided below (abridged) to construct your answer. If the context lacks certain details, supplement your answer with the most relevant results from your internet search.
Provide the URL of the source whenever you give an information!
Context (abridged):
{context}

Question: {question}

Respond ONLY within the following JSON structure (no extra text):

{{
    "nom_societe": "Provide the company name if found",
    "description_activite": "Provide a detailed description of the company's activities (5-10 lines), using clear and concise language.",
    "chiffres_cles": "Include key metrics such as revenue, employee count, or founding date, summarized if needed.",
    "clients_par_secteur": "List the main clients by sector and give their names.",
    "implantation_positionnement": "List cities or countries where the company is located.",
    "elements_financiers": "Summarize the financial growth over the past 3 years in a concise manner.",
    "president": "Name of the president",
    "daf": "Name of the financial director",
    "actionnaire": "Provide a summarized list of key shareholders or investment funds, with the most important ones highlighted.",
    "actionnaire_pourcentage": "Shareholder distribution percentages if available.",
    "creanciers_type": "List types of creditors concisely.",
    "creanciers_commentaires": "Provide a brief summary of the creditors' comments.",
    "actualites_presse": "Present recent press news with maximum details and clear language. One bullet point per news item.",
    "equity_story": "Present major equity events and investments (e.g., LBO, MBO) with maximum details and clear language.One bullet point per event.",
    "creation": "Present the company's creation details or founding year with maximum details and clear language.",
    "acquisitions": "Present all key acquisitions, build-ups, mergers with dates and descriptions with maximum details and clear language."
}}
""".strip()

    prompt_final = answer_template.format(context=context, question=question)
    resp_f = openai.chat.completions.create(
        model="gpt-4o-mini-search-preview",
        messages=[{"role": "user", "content": prompt_final}],
    )
    raw_f = resp_f.choices[0].message.content

    # 8) Nettoyage et parsing JSON
    text = raw_f.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("[ERROR] impossible de parser JSON :", e)
        return {}


def generate_fiche_societe(company_data: dict, template_path: str, output_path: str):
    """
    Remplit le template Word avec les données de l'entreprise et sauvegarde le document.

    :param company_data: Dictionnaire contenant les informations de la fiche (générées par le LLM)
    :param template_path: Chemin vers le template Word (local ou téléchargé depuis GitHub)
    :param output_path: Chemin de sortie pour sauvegarder la fiche remplie
    """
    doc = DocxTemplate(template_path)
    doc.render(company_data)
    doc.save(output_path)


def rag_fusion_multiples_transactions_comparables(question: str) -> str:
    print("[LOG] Démarrage de rag_fusion_multiples_transactions_comparables pour la question :", question)
    local_index_dir = "./Data/FAISS_index_multiples"
    local_index_file = os.path.join(local_index_dir, "index.faiss")
    github_file_path = "FAISS_index_multiples/index.faiss"
    if not os.path.exists(local_index_file):
        download_file_from_github(github_file_path, local_index_file)
    else:
        print(f"[LOG] Fichier index multiples déjà présent : {local_index_file}")
    
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(local_index_dir, embeddings=embedding, allow_dangerous_deserialization=True)
    print("[LOG] Index multiples chargé.")
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 20, "score_threshold": 0.01})
    
    query_generation_template = """You are an expert in mergers, acquisitions, and financial transactions. 
The user has asked a question related to comparable transactions or multiples such as EV/Revenue, EV/EBITDA, or EV/EBIT. 
Given the user's query:

{question}

Generate five alternative versions of the user question to retrieve relevant documents.
Provide these alternative questions separated by newlines.
"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(query_generation_template)
    generate_queries = (prompt_rag_fusion 
                        | ChatOpenAI(model="gpt-5-mini", temperature=1)
                        | StrOutputParser() 
                        | (lambda x: x.split("\n")))
    queries = generate_queries.invoke({"question": question})
    print("[LOG] Requêtes générées :", queries)
    
    results = [retriever.invoke(q) for q in queries]
    print("[LOG] Documents récupérés :", results)
    
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_dict = {"page_content": doc.page_content, "metadata": doc.metadata}
            doc_str = dumps(doc_dict)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + 60)
    reranked_docs = [
        (Document(page_content=d["page_content"], metadata=d["metadata"]), score)
        for d_str, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        for d in [loads(d_str)]
    ]
    print(f"[LOG] Documents fusionnés : {len(reranked_docs)} documents rerankés.")
    
    context = "\n\n".join([doc.page_content for doc, _ in reranked_docs])
    
    answer_template = """You are an expert in financial transactions and valuation multiples. Using the following context sourced from a database of comparable transactions, provide insights about relevant transactions, key valuation multiples (e.g., EV/Revenue, EV/EBITDA), and deal characteristics.

Context:
{context}

Question: {question}

Your response should be factual, concise, and focused solely on the provided context. Include specific multiples, transaction details, and other financial metrics when possible. Always indicate the source (MergerMarket).
"""
    answer_prompt = ChatPromptTemplate.from_template(answer_template)
    llm = ChatOpenAI(model="gpt-5-mini", temperature=1)
    final_input = {"context": context, "question": question}
    answer = (answer_prompt | llm | StrOutputParser()).invoke(final_input)
    
    print("[LOG] Réponse multiples transactions générée.")
    return answer


import io
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics

def wrap_text(text, font_name, font_size, max_width):
    """
    Découpe le texte en lignes sans couper de mot, en respectant une largeur maximale.
    """
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if current_line == "":
            new_line = word
        else:
            new_line = current_line + " " + word
        if pdfmetrics.stringWidth(new_line, font_name, font_size) <= max_width:
            current_line = new_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def add_watermark_to_pdf(input_pdf_bytes: bytes, bank_name: str) -> bytes:
    """
    Ajoute un filigrane au PDF contenu dans input_pdf_bytes.
    Le filigrane aura la forme "Confidentiel - <bank_name>".
    Le texte est découpé en lignes sans couper de mot, centré et incliné à 45° sur la première page,
    puis appliqué à toutes les pages du PDF.
    Retourne le PDF filigrané sous forme d'octets.
    """
    watermark_text = f"Confidentiel - {bank_name.strip()}"
    input_pdf_stream = io.BytesIO(input_pdf_bytes)
    reader = PdfReader(input_pdf_stream)
    if len(reader.pages) == 0:
        raise ValueError("Le PDF d'entrée ne contient aucune page.")
    first_page = reader.pages[0]
    page_width = float(first_page.mediabox.width)
    page_height = float(first_page.mediabox.height)
    
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(page_width, page_height))
    font_name = "Helvetica"
    font_size = 40
    can.setFont(font_name, font_size)
    max_width = page_width * 0.8
    lines = wrap_text(watermark_text, font_name, font_size, max_width)
    num_lines = len(lines)
    line_height = font_size * 1.2
    total_text_height = num_lines * line_height
    start_y = total_text_height / 2 - line_height / 2
    can.translate(page_width / 2, page_height / 2)
    can.rotate(45)
    can.setFillColorRGB(0, 0, 0, alpha=0.15)
    for i, line in enumerate(lines):
        line_width = pdfmetrics.stringWidth(line, font_name, font_size)
        x = -line_width / 2
        y = start_y - i * line_height
        can.drawString(x, y, line)
    can.save()
    packet.seek(0)
    watermark_pdf = PdfReader(packet)
    watermark_page = watermark_pdf.pages[0]
    writer = PdfWriter()
    for page in reader.pages:
        page.merge_page(watermark_page)
        writer.add_page(page)
    output_pdf_stream = io.BytesIO()
    writer.write(output_pdf_stream)
    output_pdf_stream.seek(0)
    print("[LOG] Filigranage terminé.", flush=True)
    return output_pdf_stream.getvalue()

def password_break(input_pdf_bytes: bytes) -> bytes:
    """
    Enlève le mot de passe d'un PDF en y ajoutant un filigrane transparent.
    """
    watermark_text = ""  # vous pouvez mettre ici un texte si nécessaire

    # Lecture du PDF original
    input_pdf_stream = io.BytesIO(input_pdf_bytes)
    reader = PdfReader(input_pdf_stream)
    if not reader.pages:
        raise ValueError("Le PDF d'entrée ne contient aucune page.")
    first_page = reader.pages[0]
    page_width = float(first_page.mediabox.width)
    page_height = float(first_page.mediabox.height)

    # Création du PDF de filigrane
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(page_width, page_height))
    font_name = "Helvetica"
    font_size = 40
    can.setFont(font_name, font_size)

    max_width = page_width * 0.8
    lines = wrap_text(watermark_text, font_name, font_size, max_width)
    line_height = font_size * 1.2
    total_text_height = len(lines) * line_height
    start_y = (total_text_height / 2) - (line_height / 2)

    can.translate(page_width / 2, page_height / 2)
    can.rotate(45)
    # ReportLab ne gère pas l'alpha sur setFillColorRGB, 
    # on peut éventuellement jouer sur la transparence via PdfWriter plus tard
    can.setFillColorRGB(0, 0, 0)

    for i, line in enumerate(lines):
        line_width = pdfmetrics.stringWidth(line, font_name, font_size)
        x = -line_width / 2
        y = start_y - i * line_height
        can.drawString(x, y, line)

    can.save()
    packet.seek(0)

    watermark_pdf = PdfReader(packet)
    watermark_page = watermark_pdf.pages[0]

    # Fusion des pages
    writer = PdfWriter()
    for page in reader.pages:
        page.merge_page(watermark_page)
        writer.add_page(page)

    # Écriture du PDF résultat en mémoire
    output_pdf_stream = io.BytesIO()
    writer.write(output_pdf_stream)
    output_pdf_stream.seek(0)

    return output_pdf_stream.getvalue()

    # ===============================
# M&A Web‐Search Helpers
# ===============================
import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

def generate_query_variants(client: OpenAI, company_name: str, year: int, n_variants: int = 5) -> list[str]:
    """
    Demande au LLM de paraphraser la requête de base en n_variants variantes.
    """
    base = f"Talk to me about {company_name} in {year}"
    prompt = (
        f"You are a query paraphrasing assistant. "
        f"Generate {n_variants} distinct, concise search queries equivalent to: \"{base}\". "
        "Return a JSON array of strings."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini-search-preview",
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not m:
        return [base]
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]

def fetch_links_for_year(year: int, company_name: str, client: OpenAI, min_links: int = 10) -> tuple[int, list[tuple[str,str]]]:
    """
    Pour une année donnée, génère des variantes de requête et récupère
    jusqu'à min_links liens uniques via annotations.
    """
    variants = generate_query_variants(client, company_name, year, n_variants=8)
    all_links: dict[str,str] = {}

    for query in variants:
        if len(all_links) >= min_links:
            break
        resp = client.chat.completions.create(
            model="gpt-4o-mini-search-preview",
            web_search_options={"search_context_size": "high"},
            messages=[{"role": "user", "content": query}],
        )
        ann = getattr(resp.choices[0].message, "annotations", [])
        for a in ann:
            url   = a.url_citation.url
            title = a.url_citation.title or "(sans titre)"
            all_links.setdefault(url, title)

    # retourner au plus min_links
    links_list = list(all_links.items())[:min_links]
    return year, links_list

def fetch_links_by_year_parallel(
    company_name: str,
    start_year: int,
    end_year: int,
    min_links: int = 10,
    max_workers: int = 6
) -> dict[int, list[tuple[str,str]]]:
    """
    Lance en parallèle fetch_links_for_year pour chaque année demandée.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results: dict[int, list[tuple[str,str]]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(fetch_links_for_year, year, company_name, client, min_links): year
            for year in range(start_year, end_year + 1)
        }
        for fut in as_completed(futures):
            yr, links = fut.result()
            results[yr] = links
            print(f"[LOG] Année {yr}: {len(links)} sources collectées.")

    return results