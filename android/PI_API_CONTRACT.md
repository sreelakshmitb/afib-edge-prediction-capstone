# Raspberry Pi → Android contract

The app expects a local Pi service. The phone does not connect directly to ESP32 BLE.
For the Wi-Fi replay integration, the ESP32 connects to the Pi's existing TCP
receiver on port 8766. The phone controls the Pi bridge on port 8080.

## Transport

- Health and resynchronization: `GET /api/v1/status`
- Session list: `GET /api/v1/sessions`
- Session detail: `GET /api/v1/sessions/{id}`
- Event stream: `WS /api/v1/events`
- Start selected replay: `POST /api/v1/replay/start` with JSON
  `{"recording_id":"POS102"}` or `{"recording_id":"NEG024"}`
- Default development address: `http://192.168.4.1:8080`

HTTP is allowed for the trusted local research network. Use authentication and TLS if the service is exposed beyond that isolated network.

## Common envelope

Every streamed event should include:

```json
{
  "type": "session.state",
  "schema_version": 1,
  "session_id": "session-001",
  "seq": 314,
  "timestamp_utc": "2026-11-14T07:00:00Z",
  "mode": "live"
}
```

`session_id` prevents data from different runs being combined. `seq` must increase within a session. The app ignores duplicate or older sequence values.

## Events

### `device.status`

```json
{
  "type": "device.status",
  "schema_version": 1,
  "session_id": "session-001",
  "seq": 1,
  "mode": "live",
  "pi_connection": "connected",
  "pcb_connection": "connected"
}
```

### `session.state`

```json
{
  "type": "session.state",
  "schema_version": 1,
  "session_id": "session-001",
  "seq": 2,
  "mode": "live",
  "phase": "collecting",
  "valid_segments": 14,
  "elapsed_seconds": 420,
  "message": "Collecting a complete 10-minute observation"
}
```

### `ecg.preview`

The preview is deliberately bounded and downsampled. The Pi must retain the full 250 Hz signal for preprocessing and inference.

```json
{
  "type": "ecg.preview",
  "schema_version": 1,
  "session_id": "session-001",
  "seq": 3,
  "mode": "live",
  "units": "mV",
  "values": [0.02, 0.04, 0.91, -0.22, 0.05]
}
```

### `signal.quality`

Recognized values: `valid`, `good`, `checking`, `low`, `poor`, `gap`, `data_gap`, and `unavailable`.

### `forecast.result`

```json
{
  "type": "forecast.result",
  "schema_version": 1,
  "session_id": "session-001",
  "forecast_id": "forecast-001",
  "seq": 314,
  "timestamp_utc": "2026-11-14T07:00:00Z",
  "mode": "replay",
  "source_kind": "recorded_ecg",
  "recording_id": "AF-SIM-07",
  "observation_start_utc": "2026-11-14T06:50:00Z",
  "observation_end_utc": "2026-11-14T07:00:00Z",
  "horizon_minutes": 20,
  "score": 0.108187,
  "threshold": null,
  "decision": "unavailable",
  "quality": "valid",
  "model_id": "frozen-v1",
  "is_demo": true
}
```

The app displays a numeric score without interpretation when `threshold` is `null`. The Pi must not emit a result until the complete trained input contract has been satisfied.

### `forecast.error`

```json
{
  "type": "forecast.error",
  "schema_version": 1,
  "session_id": "session-001",
  "seq": 315,
  "mode": "live",
  "message": "Insufficient valid R peaks for HRV features"
}
```

## Mode values

- `live`
- `replay`
- `model_validation` or `model_validation_replay`
- `ui_demo` or `fixture`

The Pi is authoritative. A phone-side view selection must not be interpreted as a command to change the PCB's physical mode.
