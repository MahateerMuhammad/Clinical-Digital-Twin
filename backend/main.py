"""
backend/main.py
───────────────
FastAPI application for the clinician backend.

    .venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8010

Scope, stated once and meant
────────────────────────────
Localhost, no authentication, in-memory sessions, single worker. It is a
development and demonstration target. It must not be exposed to a network and
must not receive real patient data — there is nothing here that would protect
either. See backend/README.md.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import routes_assistant, routes_models
from backend.service import service

#: Where the web UI will be served from in development. Comma-separated
#: override via CDT_CORS_ORIGINS.
CORS_ORIGINS = [
    o.strip() for o in os.environ.get(
        "CDT_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Once, at startup: unpickling the boosters takes seconds and doing it per
    # request would put that on every caller, including the ones that never
    # need a model.
    service.start()
    yield


app = FastAPI(
    title="Clinical Digital Twin — clinician API",
    description=(
        "Risk estimation over MIMIC-IV-trained models and a curated guideline "
        "corpus. Refuses rather than imputes: an incomplete payload returns the "
        "list of missing fields, and any generated text failing the grounding "
        "check is withheld. Localhost only; not for real patient data."),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_models.router)
app.include_router(routes_assistant.router)


@app.get("/api/health", tags=["meta"])
def health() -> Dict[str, Any]:
    """What loaded, what is available, and what the corpora actually contain.

    Reports the guideline count and how many records are clinician-reviewed,
    because "23 records, 0 reviewed" is the single most important thing a
    clinician should know before reading anything this service says.
    """
    return service.describe()
