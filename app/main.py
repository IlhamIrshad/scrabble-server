"""Scrabble board capture server.

  POST /detect          multipart image -> board state JSON
  POST /detect/raw      raw image bytes in the body (what the ESP32 will use)
  WS   /ws              send JPEG frames, get board state back per frame
  GET  /debug/{name}    annotated diagnostic image
  GET  /health
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.websockets import WebSocketDisconnect

from .config import settings
from .schemas import DetectResponse, ErrorResponse
from .vision import classify, pipeline, registration

# uvicorn configures this one, so the line actually reaches the console.
log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Build the template bank up front so misconfiguration fails at startup.

    Otherwise the first request is the one that discovers the font is missing,
    which surfaces as an opaque 500 rather than a refusal to boot.
    """
    bank = classify.bank()
    log.info("templates ready: %d letters from %s", len(bank.letters), bank.source)
    yield


app = FastAPI(
    title="Scrabble Capture",
    version="0.1.0",
    description="Board state from a single photograph.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten once the UI has a fixed origin
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_corners(raw: str | None) -> np.ndarray | None:
    """Optional manual override: "x1,y1,x2,y2,x3,y3,x4,y4"."""
    if not raw:
        return None
    try:
        vals = [float(v) for v in raw.replace(" ", "").split(",")]
    except ValueError:
        raise HTTPException(422, "corners must be 8 comma-separated numbers")
    if len(vals) != 8:
        raise HTTPException(422, "corners must be 8 comma-separated numbers")
    return np.array(vals, dtype=np.float32).reshape(4, 2)


def _run(data: bytes, corners: np.ndarray | None, debug: bool):
    try:
        image = pipeline.decode(data)
    except pipeline.DecodeError as exc:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error=str(exc), stage="decode").model_dump(),
        )
    try:
        return pipeline.detect(image, corners=corners, want_debug=debug)
    except registration.RegistrationError as exc:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(error=str(exc), stage="registration").model_dump(),
        )


@app.get("/health")
def health():
    bank = classify.bank()
    return {
        "ok": True,
        "version": app.version,
        "templates": {"source": bank.source, "letters": len(bank.letters)},
        "warp_size": settings.warp_size,
    }


@app.post("/detect", response_model=DetectResponse, responses={422: {"model": ErrorResponse}})
async def detect(
    image: UploadFile = File(..., description="JPEG or PNG of the board"),
    debug: bool = Query(False, description="Also render a diagnostic image"),
    corners: str | None = Query(None, description="Override: x1,y1,...,x4,y4"),
):
    return _run(await image.read(), _parse_corners(corners), debug)


@app.post("/detect/raw", response_model=DetectResponse, responses={422: {"model": ErrorResponse}})
async def detect_raw(
    request: Request,
    debug: bool = Query(False),
    corners: str | None = Query(None),
):
    return _run(await request.body(), _parse_corners(corners), debug)


@app.get("/debug/{name}")
def debug_image(name: str):
    if not name.endswith(".jpg") or "/" in name or ".." in name:
        raise HTTPException(400, "bad name")
    path = os.path.join(settings.debug_dir, name)
    if not os.path.exists(path):
        raise HTTPException(404, "expired or never existed")
    return FileResponse(path, media_type="image/jpeg")


@app.websocket("/ws")
async def ws(socket: WebSocket):
    """One JPEG frame in, one JSON board state out. For the ESP32 phase."""
    await socket.accept()
    try:
        while True:
            frame = await socket.receive_bytes()
            try:
                image = pipeline.decode(frame)
                result = pipeline.detect(image).model_dump()
            except (pipeline.DecodeError, registration.RegistrationError) as exc:
                result = {"ok": False, "error": str(exc)}
            await socket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        return
