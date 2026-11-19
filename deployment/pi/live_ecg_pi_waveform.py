"""Receive MAX30003 ECG over Wi-Fi and score complete 10-minute windows.

Run from ~/afib_replay alongside model/ and _training_src/.
Wire protocol: ASCII `LIVE 250 20\n`, then repeated little-endian
<uint32 sequence, uint16 count, uint16 flags, count * int32 ADC codes>.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time

import numpy as np
import onnxruntime as ort
from scipy.special import expit
from wfdb.processing import xqrs_detect

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "_training_src/model_bundles/se_last_mask_seed7/code/src"))
from afib_edge.signal import bandpass_filter
from afib_edge.hrv import compute_hrv_features
from afib_edge.hrv_preprocess import HRVFoldPreprocessor

FS = 250
MIN_AVERAGE_RATE = 245  # Allow small clock/transport drift; keep the 12 s socket watchdog.
WINDOW = 150_000
SEGMENT = 7_500
GAIN = 20
THRESHOLD = 0.3310546875
UV_PER_COUNT = 1_000_000.0 / ((1 << 17) * GAIN)
HEADER = struct.Struct("<IHH")
STATUS = ROOT / "live_status.json"
WAVEFORM_COUNTS = ROOT / "live_waveform_10s_counts.npy"
WAVEFORM_SVG = ROOT / "live_waveform_10s.svg"


def save_waveform(counts):
    """Save a raw 10-second strip and a browser-viewable voltage plot."""
    np.save(WAVEFORM_COUNTS, counts)
    uv = counts.astype(np.float64) * UV_PER_COUNT
    low, high = np.percentile(uv, [1, 99])
    pad = max((high - low) * 0.1, 1.0)
    low -= pad
    high += pad
    width, height = 1100, 360
    left, top, plot_w, plot_h = 85, 35, 980, 270
    span = high - low
    clipped = np.clip(uv, low, high)
    points = " ".join(
        f"{left + i * plot_w / (len(uv) - 1):.1f},"
        f"{top + (high - value) * plot_h / span:.1f}"
        for i, value in enumerate(clipped)
    )
    grid = "".join(
        f'<line x1="{left + second * plot_w / 10:.1f}" y1="{top}" '
        f'x2="{left + second * plot_w / 10:.1f}" y2="{top + plot_h}" '
        f'stroke="#d8e2e8"/><text x="{left + second * plot_w / 10:.1f}" '
        f'y="{top + plot_h + 22}" text-anchor="middle">{second}</text>'
        for second in range(11)
    )
    WAVEFORM_SVG.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<g font-family="sans-serif" font-size="13" fill="#253746">'
        '<text x="85" y="22">Live ECG: 10-second raw voltage strip (after 10 s settling)</text>'
        f'{grid}<text x="{left + plot_w / 2}" y="{height - 12}" text-anchor="middle">Time (s)</text>'
        f'<text x="10" y="{top + 10}">{high:.0f} µV</text>'
        f'<text x="10" y="{top + plot_h}">{low:.0f} µV</text>'
        f'<polyline fill="none" stroke="#087f8c" stroke-width="1.2" points="{points}"/>'
        '</g></svg>\n', encoding="utf-8"
    )
    print(f"Saved waveform: {WAVEFORM_SVG} (raw samples: {WAVEFORM_COUNTS})", flush=True)


def publish(**values):
    record = {"mode": "live", "updated_at_epoch": time.time(), **values}
    temp = STATUS.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temp, STATUS)
    print(json.dumps(record), flush=True)


def receive_exact(conn, count):
    data = bytearray()
    while len(data) < count:
        block = conn.recv(count - len(data))
        if not block:
            raise ConnectionError("ESP32 disconnected")
        data.extend(block)
    return bytes(data)


def receive_line(conn, limit=80):
    line = bytearray()
    while len(line) < limit:
        char = receive_exact(conn, 1)
        if char == b"\n":
            return line.decode("ascii")
        line.extend(char)
    raise ValueError("Invalid stream greeting")


class Forecast:
    def __init__(self):
        state = json.loads((ROOT / "model/hrv_preprocessor.json").read_text())
        self.preprocessor = HRVFoldPreprocessor.from_state_dict(state)
        self.session = ort.InferenceSession(
            str(ROOT / "model/forecast_model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.inputs = self.session.get_inputs()
        if len(self.inputs) != 2:
            raise ValueError("Expected two ONNX inputs: ECG and HRV")

    def score(self, raw_counts):
        if len(raw_counts) != WINDOW:
            raise ValueError("Inference requires exactly 150000 samples")
        # The replay path converts microvolts to millivolts before filtering.
        ecg = np.asarray(raw_counts, dtype=np.float64) * (UV_PER_COUNT / 1000.0)
        ecg = bandpass_filter(ecg, FS)
        peaks = np.asarray(
            xqrs_detect(sig=ecg.astype(np.float64), fs=FS, verbose=False),
            dtype=np.int64,
        )
        segments = ecg.reshape(20, SEGMENT).copy()
        hrv = np.empty((20, 6), dtype=np.float32)
        for i in range(20):
            lo, hi = i * SEGMENT, (i + 1) * SEGMENT
            local = peaks[(peaks >= lo) & (peaks < hi)] - lo
            hrv[i] = compute_hrv_features(local, FS).values
        mean = segments.mean(axis=-1, keepdims=True)
        std = segments.std(axis=-1, keepdims=True)
        segments = (segments - mean) / np.maximum(std, 1e-6)
        ecg_input = segments.reshape(1, 20, 1, SEGMENT).astype(np.float32)
        hrv_input = self.preprocessor.transform(hrv[None, :, :])
        output = self.session.run(
            None,
            {self.inputs[0].name: ecg_input, self.inputs[1].name: hrv_input},
        )[0]
        return float(expit(float(np.asarray(output).reshape(-1)[0]))), len(peaks)


def handle(conn, address, model, stride):
    conn.settimeout(12)
    greeting = receive_line(conn)
    if greeting != "LIVE 250 20":
        raise ValueError(f"Unexpected greeting: {greeting!r}")
    samples = deque(maxlen=WINDOW)
    expected_seq = 0
    total = 0
    next_score = WINDOW
    last_progress = time.monotonic()
    start = time.monotonic()
    result = {}
    pending = None
    waveform_saved = False
    publish(state="collecting", progress_seconds=0, source_ip=address[0])
    with ThreadPoolExecutor(max_workers=1) as worker:
        while True:
            seq, count, flags = HEADER.unpack(receive_exact(conn, HEADER.size))
            if count < 1 or count > 50 or flags != 0 or seq != expected_seq:
                raise ValueError(
                    f"Stream discontinuity: sequence {seq}, expected {expected_seq}, "
                    f"count {count}, flags {flags}"
                )
            payload = receive_exact(conn, count * 4)
            block = np.frombuffer(payload, dtype="<i4")
            if np.any(block < -(1 << 17)) or np.any(block >= (1 << 17)):
                raise ValueError("ADC sample outside signed 18-bit range")
            samples.extend(block.tolist())
            total += count
            expected_seq += 1
            if not waveform_saved and total >= 20 * FS:
                strip = np.fromiter(list(samples)[-10 * FS:], dtype=np.int32)
                save_waveform(strip)
                waveform_saved = True
            now = time.monotonic()
            elapsed = now - start
            observed_rate = total / elapsed if elapsed else 0.0
            # The packet sequence and the MAX30003 FIFO checks guard continuity.
            # A strict cumulative 250 Hz wall-clock deadline rejected continuous
            # streams just before the first 150,000-sample window completed.
            if total > FS * (now - start + 10):
                raise ValueError("Sample stream faster than 250 Hz")
            if elapsed > total / MIN_AVERAGE_RATE + 12:
                raise ValueError(
                    f"Sample stream too slow: {observed_rate:.2f} Hz average, "
                    f"{elapsed:.1f} s elapsed, {total} samples"
                )
            if pending is not None and pending.done():
                probability, peaks = pending.result()
                result = dict(probability=probability, threshold=THRESHOLD,
                              elevated=probability >= THRESHOLD, r_peaks=peaks)
                publish(state="result", progress_seconds=600,
                        received_samples=total, source_ip=address[0],
                        wall_elapsed_seconds=round(elapsed, 1),
                        observed_rate_hz=round(observed_rate, 2), **result)
                pending = None
            if now - last_progress >= 5:
                publish(state="analyzing" if pending else "collecting",
                        progress_seconds=min(total // FS, 600),
                        received_samples=total, source_ip=address[0],
                        wall_elapsed_seconds=round(elapsed, 1),
                        observed_rate_hz=round(observed_rate, 2), **result)
                last_progress = now
            if total >= next_score:
                if pending is None:
                    captured = np.fromiter(samples, dtype=np.int32, count=WINDOW)
                    pending = worker.submit(model.score, captured)
                    publish(state="analyzing", progress_seconds=600,
                            received_samples=total, source_ip=address[0],
                            wall_elapsed_seconds=round(elapsed, 1),
                            observed_rate_hz=round(observed_rate, 2), **result)
                else:
                    print("Skipping stride: previous inference is still running", flush=True)
                next_score += stride * FS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--stride-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.stride_seconds < 1 or args.stride_seconds > 600:
        parser.error("stride must be 1 to 600 seconds")
    model = Forecast()
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", args.port))
        server.listen(1)
        publish(state="waiting", progress_seconds=0)
        while True:
            conn, address = server.accept()
            try:
                with conn:
                    handle(conn, address, model, args.stride_seconds)
            except (OSError, ValueError, RuntimeError) as error:
                publish(state="waiting", progress_seconds=0, error=str(error))


if __name__ == "__main__":
    main()
