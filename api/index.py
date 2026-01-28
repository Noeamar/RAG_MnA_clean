import base64
import io
import json
import os
import tempfile
from datetime import datetime
from functools import lru_cache
from typing import Any

import pandas as pd
import pytz
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from pydantic import BaseModel, Field

from poc_RAG import (
    add_watermark_to_pdf,
    ensure_data_file,
    fetch_links_by_year_parallel,
    generate_fiche_societe,
    password_break,
    rag_fusion_actualites,
    rag_fusion_actualites_search_preview,
    rag_fusion_fiche_societe_to_word,
    rag_fusion_fiche_societe_to_word_websearch,
    rag_fusion_fonds,
    rag_fusion_multiples_transactions_comparables,
)

app = FastAPI()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_PATH = os.path.join(BASE_DIR, "Data", "Template - Fiche société.docx")


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)


class CompanyRequest(BaseModel):
    company_name: str = Field(..., min_length=1)
    use_web_search: bool = False


class WatermarkRequest(BaseModel):
    pdf_base64: str = Field(..., min_length=1)
    bank_name: str = Field(..., min_length=1)


class BreakPdfRequest(BaseModel):
    pdf_base64: str = Field(..., min_length=1)


class PressLinksRequest(BaseModel):
    company: str = Field(..., min_length=1)
    start_year: int
    end_year: int
    min_links: int = 10


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3)
    job: str = Field(..., min_length=2)


@lru_cache(maxsize=1)
def get_drive_service():
    sa_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT_JSON secret.")
    sa_info = json.loads(sa_json)
    credentials = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=credentials)


def load_user_data(file_id: str) -> pd.DataFrame:
    try:
        drive_service = get_drive_service()
        meta = drive_service.files().get(fileId=file_id, fields="mimeType").execute()
        mime = meta.get("mimeType", "")
        if mime == "application/vnd.google-apps.spreadsheet":
            request = drive_service.files().export_media(
                fileId=file_id, mimeType="text/csv"
            )
        else:
            request = drive_service.files().get_media(fileId=file_id)

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        df = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8"))
    except Exception:
        df = pd.DataFrame(columns=["email", "job", "last_connection"])

    if "last_connection" not in df.columns:
        df["last_connection"] = pd.NA
    return df


def save_user_data(df: pd.DataFrame, file_id: str):
    drive_service = get_drive_service()
    fh = io.BytesIO()
    fh.write(df.to_csv(index=False).encode("utf-8"))
    fh.seek(0)
    media = MediaIoBaseUpload(fh, mimetype="text/csv", resumable=True)
    drive_service.files().update(fileId=file_id, media_body=media).execute()


def build_company_docx(company_data: dict, output_name: str) -> bytes:
    ensure_data_file("Data/Template - Fiche société.docx")
    if not os.path.exists(TEMPLATE_PATH):
        raise RuntimeError("Template Word introuvable.")
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
        generate_fiche_societe(company_data, TEMPLATE_PATH, tmp.name)
        tmp.seek(0)
        doc_bytes = tmp.read()
    if not doc_bytes:
        raise RuntimeError("Fiche Word vide.")
    return doc_bytes


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/news")
def news(req: QuestionRequest):
    return {"answer": rag_fusion_actualites(req.question)}


@app.post("/news-web")
def news_web(req: QuestionRequest):
    return {"answer": rag_fusion_actualites_search_preview(req.question)}


@app.post("/funds")
def funds(req: QuestionRequest):
    return {"answer": rag_fusion_fonds(req.question)}


@app.post("/comparables")
def comparables(req: QuestionRequest):
    return {"answer": rag_fusion_multiples_transactions_comparables(req.question)}


@app.post("/company-data")
def company_data(req: CompanyRequest):
    question = f"Fournis-moi une fiche détaillée pour l'entreprise {req.company_name}."
    if req.use_web_search:
        data = rag_fusion_fiche_societe_to_word_websearch(question)
    else:
        data = rag_fusion_fiche_societe_to_word(question)
    return {"data": data}


@app.post("/company-docx")
def company_docx(req: CompanyRequest):
    question = f"Fournis-moi une fiche détaillée pour l'entreprise {req.company_name}."
    if req.use_web_search:
        data = rag_fusion_fiche_societe_to_word_websearch(question)
    else:
        data = rag_fusion_fiche_societe_to_word(question)

    doc_bytes = build_company_docx(data, f"{req.company_name}_fiche_societe.docx")
    filename = f"{req.company_name}_fiche_societe.docx".replace(" ", "_")
    return StreamingResponse(
        io.BytesIO(doc_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/press-links")
def press_links(req: PressLinksRequest):
    if req.start_year > req.end_year:
        raise HTTPException(status_code=400, detail="start_year must be <= end_year")
    links = fetch_links_by_year_parallel(
        req.company,
        req.start_year,
        req.end_year,
        min_links=req.min_links,
    )
    return {"links_by_year": links}


@app.post("/watermark")
def watermark(req: WatermarkRequest):
    try:
        pdf_bytes = base64.b64decode(req.pdf_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF") from exc
    result = add_watermark_to_pdf(pdf_bytes, req.bank_name)
    encoded = base64.b64encode(result).decode("utf-8")
    return {"pdf_base64": encoded}


@app.post("/break-pdf")
def break_pdf(req: BreakPdfRequest):
    try:
        pdf_bytes = base64.b64decode(req.pdf_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF") from exc
    result = password_break(pdf_bytes)
    encoded = base64.b64encode(result).decode("utf-8")
    return {"pdf_base64": encoded}


@app.post("/register")
def register(req: RegisterRequest):
    file_id = os.getenv("GDRIVE_USERS_FILE_ID")
    if not file_id:
        raise HTTPException(status_code=500, detail="Missing GDRIVE_USERS_FILE_ID.")

    df = load_user_data(file_id)
    tz = pytz.timezone("Europe/Paris")
    now = datetime.now(tz).isoformat(timespec="seconds")

    if req.email in df["email"].values:
        df.loc[df["email"] == req.email, "job"] = req.job
        df.loc[df["email"] == req.email, "last_connection"] = now
    else:
        new_row = pd.DataFrame(
            [{"email": req.email, "job": req.job, "last_connection": now}]
        )
        df = pd.concat([df, new_row], ignore_index=True)

    save_user_data(df, file_id)
    return JSONResponse({"status": "ok"})

