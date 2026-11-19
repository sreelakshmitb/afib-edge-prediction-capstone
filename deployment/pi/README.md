# Raspberry Pi deployment

Use a Pi 4B with an OS/Python combination supported by ONNX Runtime wheels (64-bit Raspberry Pi OS is recommended). Model execution is CPU-only. The final Pi package versions were not captured; requirements are installation ranges, not a lockfile.

From a cloned repository:

```bash
python3 scripts/verify_release.py
python3 scripts/prepare_pi.py --destination ~/afib_replay
cd ~/afib_replay
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The preparer copies exact model bytes and the original signal/HRV source into the `_training_src/` layout expected by the working scripts. Run `python scripts/verify_frozen_model.py` from the repository in an environment with these dependencies.

## Live mode

ESP32 off initially. Terminal 1:

```bash
cd ~/afib_replay
source .venv/bin/activate
python -u live_ecg_pi_waveform.py
```

Terminal 2:

```bash
cd ~/afib_replay
source .venv/bin/activate
python -u pi_app_bridge.py
```

Turn on ESP32 with live firmware. Connect Android to the Pi, turn Presentation simulation off, choose Live testing, and start. It takes 150,000 valid samples (ten minutes at 250 Hz) before the first forecast; subsequent inference uses a rolling window. No forecast is available during warm-up.

## Replay through Android

Turn off ESP32, stop the live receiver and bridge with Ctrl+C, then:

```bash
cd ~/afib_replay
source .venv/bin/activate
rm -f live_status.json
python -u pi_app_bridge.py
```

Turn on ESP32 with the project's Wi-Fi SD replay firmware. Connect the app, choose replay and POS102 or NEG024. The bridge starts reception and inference. Do not also start a manual receiver on port 8766.

## Manual replay

Stop the bridge first. With the matching SD replay firmware:

```bash
python -u wifi_replay_receiver.py P
python -u demo_replay.py POS102
# For the negative case, receive N then run NEG024 instead.
```

Each file must be 600,000 bytes: 150,000 little-endian int32 microvolt samples. The receiver uses a temporary file, checks size/SHA-256 and installs it atomically. It can replace a previous valid capture; routine reruns do not require deleting captures.

| Port | Purpose |
| --- | --- |
| 8766 TCP | SD replay transfer |
| 8767 TCP | Live sample stream |
| 8080 HTTP/WebSocket | Android bridge, /api/v1/status and /api/v1/events |

Use the same local network for ESP32, Pi and phone; set the correct Pi IP and hotspot credentials in firmware. The bridge is a local-network prototype with no authentication; do not expose its port to the Internet. See [Android API contract](../../android/PI_API_CONTRACT.md).

POS102/NEG024 and the final ESP32 replay sketch are absent from this recovered source release. Recover those from the working device/SD before rebuilding that part. The live firmware is separate. Record final Pi latency, peak RAM and long-duration runs before reporting deployment benchmarks.
