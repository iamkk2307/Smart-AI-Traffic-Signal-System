"""
Application entrypoint.

The control loop runs as a lifespan-managed task, not as a side effect of the
first request, so the signal keeps running whether or not anyone has the
dashboard open. The WebSocket hub is bounded and guarded by a lock: the
naive version — a bare list mutated from both the receive handler and the
broadcast task — drops clients under concurrent disconnects, and an
unbounded list is a trivial way to exhaust the process.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import analytics, ml, signals, state
from app.config import get_settings
from app.db import init_db
from app.services.runtime import get_runtime

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smart-traffic")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_WS_CLIENTS = 64
BROADCAST_INTERVAL = 1.0


class Hub:
    """Fan-out for live state. One lock, one bounded set, no surprises."""

    def __init__(self, limit: int = MAX_WS_CLIENTS) -> None:
        self.limit = limit
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def join(self, websocket: WebSocket) -> bool:
        async with self._lock:
            if len(self._clients) >= self.limit:
                return False
            self._clients.add(websocket)
            return True

    async def leave(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def broadcast(self, payload: str) -> None:
        async with self._lock:
            targets = list(self._clients)
        dead = []
        for client in targets:
            try:
                await client.send_text(payload)
            except Exception:
                dead.append(client)
        if dead:
            async with self._lock:
                for client in dead:
                    self._clients.discard(client)

    @property
    def size(self) -> int:
        return len(self._clients)


hub = Hub()


async def _broadcast_loop() -> None:
    runtime = get_runtime()
    while True:
        try:
            await asyncio.sleep(BROADCAST_INTERVAL)
            if hub.size == 0:
                continue
            await hub.broadcast(
                json.dumps({"type": "state", "data": runtime.state()}, default=str)
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("broadcast failed")
            await asyncio.sleep(BROADCAST_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    runtime = get_runtime()
    runtime.start()
    broadcaster = asyncio.create_task(_broadcast_loop())
    logger.info("ready on http://%s:%s", settings.api_host, settings.api_port)
    try:
        yield
    finally:
        broadcaster.cancel()
        try:
            await broadcaster
        except asyncio.CancelledError:
            pass
        await runtime.stop()
        logger.info("stopped")


app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    description=(
        "Adaptive traffic signal control with a verified safety interlock, "
        "explainable phase selection, arrival forecasting measured against a "
        "persistence baseline, and a live operations console."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Operator-Token"],
)

app.include_router(state.router, prefix="/api", tags=["State"])
app.include_router(signals.router, prefix="/api/signals", tags=["Operator"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(ml.router, prefix="/api/ml", tags=["Forecasting"])


@app.get("/api/health", tags=["State"])
def health() -> dict:
    runtime = get_runtime()
    return {
        "service": settings.app_name,
        "status": "faulted" if runtime.controller.faulted else "ok",
        "control_mode": runtime.controller.mode,
        "sensor_mode": settings.sensor_mode,
        "cycle": runtime.controller.cycle,
        "uptime_sim_seconds": round(runtime.controller.t, 1),
        "interlock_events": len(runtime.interlock.events),
        "ws_clients": hub.size,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    if not await hub.join(websocket):
        await websocket.close(code=1013, reason="too many clients")
        return
    try:
        await websocket.send_text(
            json.dumps({"type": "state", "data": get_runtime().state()}, default=str)
        )
        while True:
            # The client has nothing to tell us; this read exists to notice
            # the disconnect. Pushes come from the broadcast task.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("websocket closed unexpectedly", exc_info=True)
    finally:
        await hub.leave(websocket)


@app.get("/", include_in_schema=False)
def dashboard():
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(index, media_type="text/html")
    return JSONResponse(
        {
            "service": settings.app_name,
            "dashboard": "web/index.html is missing",
            "api": "/docs",
        },
        status_code=404,
    )


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
