"""Saved games: the board documents and the photographs they were read from.

One folder per game on disk, holding the document and every frame that produced it:

    games/<id>/game.json
    games/<id>/frames/t3-board.jpg

Files rather than a database, because a game *is* a document, the whole format was
designed to be hand-editable, and a folder you can open and read beats a row you have
to query. It also means a game survives this program being rewritten.

The sign-in here is a doorbell, not a lock. It keeps a stray browser tab from writing
over a game in progress; it stops nobody who can reach the machine, because the token
lives in memory and everything is served over plain HTTP on localhost. Do not put this
on a network and believe the games are protected.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile

from .config import settings

router = APIRouter()

# Tokens handed out by /auth/login. In memory on purpose: restarting the server signs
# everybody out, which is the right default for something with one user.
_tokens: set[str] = set()

SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _root() -> Path:
    path = Path(settings.games_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def slug(text: str) -> str:
    """A filesystem-safe name. Never empty, never a path, never a parent directory."""
    cleaned = SAFE.sub("-", (text or "").strip()).strip("-._")
    return cleaned[:64] or "game"


def _dir(game_id: str) -> Path:
    """The folder for a game, refusing anything that could climb out of the root."""
    safe = slug(game_id)
    if safe != game_id:
        raise HTTPException(400, f"bad game id: {game_id!r}")
    return _root() / safe


def require_auth(authorization: str = Header(default="")) -> None:
    token = authorization[7:] if authorization.lower().startswith("bearer ") else ""
    if token not in _tokens:
        raise HTTPException(401, "sign in first")


# -- signing in --------------------------------------------------------------


@router.post("/auth/login")
def login(body: dict):
    """Exchange a username and password for a token.

    Compared with `secrets.compare_digest` so that a wrong password takes the same time
    as a right one. That is the only part of this worth calling security, and it is
    here because it costs nothing - see the note at the top of this file for what this
    does not protect against.
    """
    user = str(body.get("username", ""))
    password = str(body.get("password", ""))
    ok = (secrets.compare_digest(user, settings.ui_username)
          and settings.ui_password != ""
          and secrets.compare_digest(password, settings.ui_password))
    if not ok:
        raise HTTPException(401, "wrong username or password")
    token = secrets.token_urlsafe(24)
    _tokens.add(token)
    return {"token": token, "username": user}


@router.post("/auth/logout")
def logout(authorization: str = Header(default="")):
    token = authorization[7:] if authorization.lower().startswith("bearer ") else ""
    _tokens.discard(token)
    return {"ok": True}


@router.get("/auth/check")
def check(_: None = Depends(require_auth)):
    return {"ok": True}


@router.get("/auth/configured")
def configured():
    """Whether a password has been set at all, so the page can say so plainly.

    Without this a misconfigured server looks exactly like a mistyped password, and the
    person at the keyboard tries their password again instead of setting one.
    """
    return {"configured": settings.ui_password != "", "username": settings.ui_username}


# -- games -------------------------------------------------------------------


def _summary(folder: Path) -> dict | None:
    doc_path = folder / "game.json"
    if not doc_path.exists():
        return None
    try:
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A half-written or hand-mangled game should not take the whole list down with
        # it; it is listed as broken so it can be found and fixed.
        return {"id": folder.name, "broken": True, "sessionId": folder.name,
                "turns": 0, "tiles": 0, "frames": 0,
                "updatedAt": doc_path.stat().st_mtime}
    turns = doc.get("turns") or []
    last = turns[-1] if turns else {}
    frames = folder / "frames"
    return {
        "id": folder.name,
        "broken": False,
        "sessionId": doc.get("sessionId", folder.name),
        "players": doc.get("players", []),
        "turns": len(turns),
        "tiles": len(last.get("cells", [])),
        "frames": len(list(frames.glob("*"))) if frames.is_dir() else 0,
        "updatedAt": doc_path.stat().st_mtime,
    }


@router.get("/games")
def list_games(_: None = Depends(require_auth)):
    out = []
    for folder in sorted(_root().iterdir()):
        if not folder.is_dir():
            continue
        item = _summary(folder)
        if item:
            out.append(item)
    out.sort(key=lambda g: g["updatedAt"], reverse=True)
    return {"games": out}


@router.get("/games/{game_id}")
def get_game(game_id: str, _: None = Depends(require_auth)):
    path = _dir(game_id) / "game.json"
    if not path.exists():
        raise HTTPException(404, f"no game called {game_id!r}")
    return json.loads(path.read_text(encoding="utf-8"))


@router.put("/games/{game_id}")
def save_game(game_id: str, doc: dict, _: None = Depends(require_auth)):
    """Write a game, replacing whatever was there.

    Written to a temporary file and moved into place, so a crash halfway through leaves
    the previous game intact rather than a truncated one. A game is the only record of
    an afternoon's play and there is no second copy.
    """
    folder = _dir(game_id)
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / "game.json"
    tmp = folder / "game.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, final)
    return {"ok": True, "id": folder.name, "savedAt": final.stat().st_mtime}


@router.delete("/games/{game_id}")
def delete_game(game_id: str, _: None = Depends(require_auth)):
    folder = _dir(game_id)
    if not folder.exists():
        raise HTTPException(404, f"no game called {game_id!r}")
    shutil.rmtree(folder)
    return {"ok": True}


# -- the photographs a game was read from ------------------------------------


@router.post("/games/{game_id}/frames/{turn}")
async def add_frame(
    game_id: str, turn: int, image: UploadFile = File(...),
    _: None = Depends(require_auth),
):
    """Keep the photograph a turn was read from, beside the game it produced.

    Worth storing even though the board state is what gets used: when a transcript is
    questioned weeks later, the picture is the only thing that can settle it, and by
    then the board has been packed away.
    """
    folder = _dir(game_id) / "frames"
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(image.filename or "frame.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}:
        raise HTTPException(400, f"not an image this keeps: {suffix}")
    name = f"t{int(turn):03d}-{int(time.time())}{suffix}"
    data = await image.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "image is larger than the upload limit")
    (folder / name).write_bytes(data)
    return {"ok": True, "name": name, "url": f"/games/{_dir(game_id).name}/frames/{name}"}


@router.get("/games/{game_id}/frames")
def list_frames(game_id: str, _: None = Depends(require_auth)):
    folder = _dir(game_id) / "frames"
    if not folder.is_dir():
        return {"frames": []}
    names = sorted(p.name for p in folder.iterdir() if p.is_file())
    return {"frames": [{"name": n, "turn": _turn_of(n),
                        "url": f"/games/{_dir(game_id).name}/frames/{n}"} for n in names]}


def _turn_of(name: str) -> int | None:
    m = re.match(r"t(\d+)-", name)
    return int(m.group(1)) if m else None


@router.get("/games/{game_id}/frames/{name}")
def get_frame(game_id: str, name: str):
    """Served without a token, so an <img> tag can reach it.

    A browser will not put an Authorization header on an image request, and the
    alternative - fetching every picture as a blob - buys nothing here: the token is
    already only a doorbell, and the file name has to be guessed anyway.
    """
    from fastapi.responses import FileResponse

    if slug(name) != name:
        raise HTTPException(400, "bad frame name")
    path = _dir(game_id) / "frames" / name
    if not path.exists():
        raise HTTPException(404, "no such frame")
    return FileResponse(path)
