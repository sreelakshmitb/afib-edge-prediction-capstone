"""Pi HTTP/WebSocket bridge for live ECG and the verified Wi-Fi SD replay demo.

Place beside live_ecg_pi_waveform.py, wifi_replay_receiver.py and demo_replay.py.
Run with the same virtual environment as those scripts: python -u pi_app_bridge.py
The live receiver runs in its own terminal and writes live_status.json.
"""

import asyncio
from datetime import datetime, timezone
import json
import re
import sys
import time
import uuid
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parent
PORT = 8080
CHOICES = {"POS102": "P", "NEG024": "N"}
THRESHOLD = 0.3310546875  # frozen validation operating point in demo_replay.py
LIVE_STATUS = ROOT / "live_status.json"
LIVE_STALE_SECONDS = 20


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Bridge:
    def __init__(self):
        self.clients = set()
        self.session = "idle"
        self.seq = 0
        self.recording = None
        self.events = []
        self.task = None
        self.process = None
        self.mode = "replay"
        self.mode_lock = None  # Physical ESP32 MODE button is authoritative when present.
        self.live_last_updated = None
        self.live_last_total = 0
        self.live_last_state = ""
        self.reset(None, "replay")

    def reset(self, recording, mode="replay"):
        self.session = uuid.uuid4().hex
        self.seq = 0
        self.mode = mode
        self.recording = recording
        self.events = []
        self.emit("device.status", pi_connection="connected", pcb_connection="disconnected")
        if mode == "replay":
            self.emit("session.state", phase="ready", valid_segments=0,
                      elapsed_seconds=0, message="Choose POS102 or NEG024 to start Wi-Fi replay")

    def emit(self, kind, **fields):
        self.seq += 1
        event = dict(type=kind, schema_version=1, session_id=self.session,
                     seq=self.seq, timestamp_utc=utc_now(), mode=self.mode,
                     source_kind="live_ecg" if self.mode == "live" else "recorded_ecg",
                     recording_id=self.recording, **fields)
        self.events.append(event)
        if len(self.events) > 100:
            self.events.pop(0)
        for ws in tuple(self.clients):
            asyncio.create_task(self.send(ws, event))
        return event

    async def send(self, ws, event):
        if not ws.closed:
            try:
                await ws.send_json(event)
            except ConnectionError:
                self.clients.discard(ws)

    async def run_script(self, filename, arg, on_line):
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", str(ROOT / filename), arg,
            cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        output = []
        try:
            async for raw in self.process.stdout:
                line = raw.decode("utf-8", "replace").strip()
                print(line, flush=True)
                output.append(line)
                on_line(line)
            code = await self.process.wait()
            if code != 0:
                raise RuntimeError(f"{filename} exited {code}: {'; '.join(output[-3:])}")
            return output
        finally:
            self.process = None

    async def run_replay(self, name):
        try:
            self.emit("session.state", phase="collecting", valid_segments=0,
                      elapsed_seconds=0, message=f"Waiting for ESP32 to send {name}.BIN")

            def received(line):
                if line.startswith("Connected from"):
                    self.emit("device.status", pi_connection="connected", pcb_connection="connected")
                match = re.fullmatch(r"Received (\d+)/600000", line)
                if match:
                    byte_count = min(int(match.group(1)), 600000)
                    segments = min(byte_count // 30000, 20)
                    self.emit("session.state", phase="collecting", received_segments=segments,
                              elapsed_seconds=segments * 30,
                              message=f"Received {byte_count:,}/600,000 bytes of prerecorded ECG")

            lines = await self.run_script("wifi_replay_receiver.py", CHOICES[name], received)
            if not any(line.startswith("PASS:") for line in lines):
                raise RuntimeError("Capture did not pass integrity verification")
            self.emit("device.status", pi_connection="connected", pcb_connection="disconnected")
            self.emit("session.state", phase="processing", valid_segments=20,
                      elapsed_seconds=600, message="Checksum passed; preprocessing and scoring")
            output = await self.run_script("demo_replay.py", name, lambda line: None)
            score_line = next((line for line in output if "probability =" in line), None)
            match = re.search(r"probability = (0(?:\.\d+)?|1(?:\.0+)?)", score_line or "")
            if not match or not any("capture checksum PASS" in line for line in output):
                raise RuntimeError("Inference did not return a verified probability")
            probability = float(match.group(1))
            issued = utc_now()
            # The recording is a 10-minute *historical* window. Do not fabricate
            # acquisition timestamps from the current transfer's wall clock.
            self.emit("forecast.result", forecast_id=f"{self.session}-{name}",
                      score=probability, threshold=THRESHOLD,
                      decision="elevated" if probability >= THRESHOLD else "below_threshold",
                      quality="valid", horizon_minutes=20, model_id="frozen-onnx-v1",
                      observation_start_utc="", observation_end_utc="",
                      recording_sha256=next((line.split("SHA-256: ")[1] for line in lines
                                             if "SHA-256: " in line), ""),
                      is_demo=True, processed_at_utc=issued)
        except Exception as exc:
            print(f"Replay failed: {exc}", file=sys.stderr, flush=True)
            self.emit("forecast.error", message=str(exc))

    def live_snapshot(self, record):
        """Translate one atomic receiver status write into app events."""
        updated = record.get("updated_at_epoch")
        state = record.get("state")
        if record.get("mode") != "live" or not isinstance(updated, (int, float)) or state not in {
            "waiting", "collecting", "analyzing", "result"
        }:
            return
        if abs(time.time() - updated) > LIVE_STALE_SECONDS or updated == self.live_last_updated:
            return
        total = int(record.get("received_samples", 0))
        new_capture = (self.mode != "live" or
                       (state == "collecting" and self.live_last_state == "waiting") or
                       (state == "collecting" and total < self.live_last_total))
        if new_capture:
            self.reset(None, "live")
        self.live_last_updated = updated
        self.live_last_total = total
        self.live_last_state = state
        connected = state != "waiting"
        self.emit("device.status", pi_connection="connected",
                  pcb_connection="connected" if connected else "disconnected")
        phase = "processing" if state == "analyzing" else "ready" if state == "waiting" else "collecting"
        seconds = min(total // 250, 600) if connected else 0
        message = (str(record.get("error")) if state == "waiting" and record.get("error")
                   else "Waiting for ESP32 live ECG" if state == "waiting"
                   else "Analyzing the 10-minute ECG window" if state == "analyzing"
                   else "Live ECG received; continuing rolling analysis" if seconds == 600
                   else "Collecting the 10-minute live ECG observation")
        self.emit("session.state", phase=phase, received_segments=min(seconds // 30, 20),
                  elapsed_seconds=seconds, message=message)
        if state == "result" and "probability" in record:
            probability = float(record["probability"])
            threshold = float(record.get("threshold", THRESHOLD))
            self.emit("forecast.result", forecast_id=f"{self.session}-{updated}",
                      score=probability, threshold=threshold,
                      decision="elevated" if probability >= threshold else "below_threshold",
                      quality="checking", horizon_minutes=20, model_id="frozen-onnx-v1",
                      observation_start_utc="", observation_end_utc="",
                      r_peaks=record.get("r_peaks"), is_demo=True)

    async def watch_live(self):
        stale_reported = False
        while True:
            await asyncio.sleep(0.8)
            if (self.task is not None and not self.task.done()) or self.mode_lock == "replay":
                continue
            try:
                record = json.loads(LIVE_STATUS.read_text())
                fresh = abs(time.time() - record.get("updated_at_epoch", 0)) <= LIVE_STALE_SECONDS
                if fresh:
                    stale_reported = False
                    self.live_snapshot(record)
                elif self.mode == "live" and not stale_reported:
                    self.emit("device.status", pi_connection="connected", pcb_connection="disconnected")
                    self.emit("session.state", phase="paused", received_segments=0,
                              elapsed_seconds=0, message="Live receiver status stopped updating")
                    stale_reported = True
            except (OSError, ValueError, TypeError):
                pass


bridge = Bridge()


async def device_mode(request):
    """ESP32 announces SW3 mode transitions to keep app readiness in sync."""
    if bridge.task is not None and not bridge.task.done():
        raise web.HTTPConflict(text="Replay transfer is still finishing")
    try:
        mode = (await request.json())["mode"]
    except (ValueError, KeyError, TypeError):
        raise web.HTTPBadRequest(text="Expected mode live or replay")
    if mode not in {"live", "replay"}:
        raise web.HTTPBadRequest(text="Expected mode live or replay")
    bridge.mode_lock = mode
    if mode != bridge.mode:
        bridge.reset(None, mode)
        if mode == "live":
            bridge.emit("session.state", phase="ready", valid_segments=0,
                        elapsed_seconds=0, message="Waiting for ESP32 live ECG")
    return web.json_response({"mode": bridge.mode})


async def status(request):
    return web.json_response({"status": "online", "session_id": bridge.session,
                              "busy": bridge.task is not None and not bridge.task.done(),
                              "events": bridge.events[-5:]})


async def start(request):
    if bridge.task is not None and not bridge.task.done():
        raise web.HTTPConflict(text="A replay is already running")
    try:
        name = (await request.json())["recording_id"]
    except (ValueError, KeyError, TypeError):
        raise web.HTTPBadRequest(text="Expected recording_id POS102 or NEG024")
    if name not in CHOICES:
        raise web.HTTPBadRequest(text="Expected recording_id POS102 or NEG024")
    bridge.reset(name, "replay")
    bridge.task = asyncio.create_task(bridge.run_replay(name))
    return web.json_response({"session_id": bridge.session, "recording_id": name}, status=202)


async def events(request):
    ws = web.WebSocketResponse(heartbeat=15)
    await ws.prepare(request)
    bridge.clients.add(ws)
    for event in bridge.events:
        await ws.send_json(event)
    try:
        async for _ in ws:
            pass
    finally:
        bridge.clients.discard(ws)
    return ws


def main():
    app = web.Application()
    async def start_watcher(app):
        task = asyncio.create_task(bridge.watch_live())
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    app.cleanup_ctx.append(start_watcher)
    app.router.add_get("/api/v1/status", status)
    app.router.add_get("/api/v1/events", events)
    app.router.add_post("/api/v1/replay/start", start)
    app.router.add_post("/api/v1/device/mode", device_mode)
    web.run_app(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
