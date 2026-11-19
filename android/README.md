# AFib Forecast Android — Wi-Fi replay integration

## Run the integrated replay

Copy `pi_app_bridge.py` into `/home/pi-user/afib_replay/` beside the existing
`wifi_replay_receiver.py` and `demo_replay.py`. On the Raspberry Pi:

```bash
cd ~/afib_replay
source .venv/bin/activate
python -m pip install aiohttp
python -u pi_app_bridge.py
```

Keep this terminal open. Do not separately start `wifi_replay_receiver.py` or
`demo_replay.py`: the bridge starts each in order when the app requests a replay.
To find the Pi address, run `hostname -I` in another terminal. Keep the Pi,
ESP32, and Android phone on the same hotspot.

Build/install this Android project with Android Studio or `gradlew.bat assembleDebug`.
In the app's connection settings, set `http://<PI_IP>:8080`, turn off
**Presentation simulation**, and connect. Choose **AF simulation**, select
**POS102** or **NEG024**, then tap **Start AF simulation**. The Pi begins waiting
for the ESP32 on TCP port 8766; power/reset the ESP32 as needed for its existing
Wi-Fi sender to connect. The app shows transferred recording progress and the
verified ONNX score. Use the other recording to repeat.

You can first check `http://<PI_IP>:8080/api/v1/status` in the phone browser.
The bridge is meant only for an isolated trusted hotspot: it listens on port
8080 and accepts replay-start requests from devices on that network.

POS102 is a prerecorded forecast-positive example; NEG024 is a prerecorded
forecast-negative example. The app labels both as replay. Transfer progress is
the amount of a historical ten-minute waveform received, not ten minutes of
live observation. The model runs only after the 600,000-byte capture passes
SHA-256 verification.

The replay transfer can finish in seconds because the stored ten-minute ECG is
sent faster than its original recording rate. The app labels the timer as
**recorded ECG duration** and identifies the replay as accelerated. A live test
requires a genuine ten minutes of continuous acquisition. The connection panel
closes after a successful Pi connection; the top refresh control reconnects to
the Pi. The replay readiness checklist needs the Pi and replay mode, while the
live checklist shows the Pi, ECG module, and signal source.

On a successful connection the home button turns green and says **Connection
successful**. It still opens connection settings when tapped. The header's
refresh icon rotates when tapped and reconnects to the Pi.

Native Android research-prototype UI for the **AI-Based Single-Lead ECG Monitoring for Early Atrial Fibrillation Prediction Using Edge Computing** project.

## What is implemented

- Warm, high-contrast Jetpack Compose interface designed for adults aged 50+
- Automatic **Live signal** or **AF simulation** mode detection from the Raspberry Pi
- Persistent, prominent labels for prerecorded, model-validation, and UI-demo data
- Calm, minimal home screen with separate Live Test and AF Simulation readiness
- Dedicated running-test experience with an immersive timer and 20-segment observation progress
- Observation details remain hidden until a test is deliberately started; no ECG preview is shown
- Separate completion/result screen plus recent and full session history
- Safe forecast presentation: no score during warm-up and no risk category without a validated threshold
- Pi health check at `GET /api/v1/status`
- Pi WebSocket stream at `WS /api/v1/events`
- Sequence and session guards to prevent duplicate or mixed-session events
- Session history, project explanation, safety guidance, and editable Pi address
- In-app physical-device guide and threshold-triggered emergency-contact SMS preparation
- Custom-drawn interface symbols; no Lucide or Material icon pack
- No gradients

## Run

Open the project in Android Studio, sync Gradle, and run the `app` configuration on a Pixel 7 or compatible device. The minimum supported Android version is API 26. The project targets API 36.

The app opens in **Presentation simulation** mode for interface review. Tap the connection symbol in the upper-right corner, disable **Presentation simulation**, enter the Pi address, and choose **Connect equipment** for real events. Live Testing and AF Simulation can be selected in the app, while the green mode dot reflects the physical mode reported by the Raspberry Pi. A running live session shows **View live status** and a completed one shows **View live result**. If the phone temporarily loses its Pi stream, **View live status** stays available, keeps the last known timer, and resumes with new Pi events after reconnection. Completed sessions are saved on the phone and remain in History after the app closes. The emergency contact icon at the top opens contact number settings; Pi settings remain under the connection icon. An elevated score appears in red; the emergency contact button opens the phone's SMS app for review and sending. Pre-recorded replays are explicitly labeled as simulations.

Default Pi address:

```text
http://192.168.4.1:8080
```

## Build

```powershell
.\gradlew.bat assembleDebug
```

The debug APK is generated at `app/build/outputs/apk/debug/app-debug.apk`.

## Safety boundary

This app does not run an independent model. It displays the Raspberry Pi's authoritative state and results. It is a research demonstration, not a medical device, diagnosis, present-AF detector, treatment guide, or emergency alert system.

For body-connected recording, use only an isolated battery-powered ECG board. Do not charge or connect USB equipment while electrodes are attached.

See [PI_API_CONTRACT.md](PI_API_CONTRACT.md) for the implemented transport contract.
