package com.afibforecast.app.data

import com.afibforecast.app.model.*
import org.json.JSONObject

object PiEventParser {
    fun apply(jsonText: String, previous: MonitorState): MonitorState {
        val root = JSONObject(jsonText)
        val sessionId = root.optString("session_id", previous.sessionId)
        val seq = root.optLong("seq", previous.sequence)
        if (sessionId == previous.sessionId && seq in 1..previous.sequence) return previous

        val mode = root.optString("mode").toMode(previous.mode)
        val base = if (sessionId != previous.sessionId) MonitorState() else previous
        val common = base.copy(
            sessionId = sessionId,
            sequence = seq,
            mode = mode,
            recordingId = root.optString("recording_id").takeIf { it.isNotBlank() } ?: base.recordingId,
            lastPacketAt = "just now",
        )
        return when (root.optString("type")) {
            "device.status" -> common.copy(
                piConnection = root.optString("pi_connection").toConnection(common.piConnection),
                pcbConnection = root.optString("pcb_connection").toConnection(common.pcbConnection),
            )
            "session.state" -> common.copy(
                phase = root.optString("phase").toPhase(common.phase),
                acceptedSegments = root.optInt("received_segments",
                    root.optInt("valid_segments", common.acceptedSegments)).coerceIn(0, 20)
                    .let { if (mode == SourceMode.LIVE && sessionId == previous.sessionId &&
                        root.optString("phase") == "paused") maxOf(it, previous.acceptedSegments) else it },
                elapsedSeconds = root.optInt("elapsed_seconds", common.elapsedSeconds).coerceIn(0, 600)
                    .let { if (mode == SourceMode.LIVE && sessionId == previous.sessionId &&
                        root.optString("phase") == "paused") maxOf(it, previous.elapsedSeconds) else it },
                message = root.optString("message", common.message),
            )
            "signal.quality" -> common.copy(
                signalQuality = root.optString("quality").toQuality(),
                message = root.optString("message", common.message),
            )
            "ecg.preview" -> {
                val values = root.optJSONArray("values")
                val points = if (values == null) emptyList() else List(values.length()) { values.optDouble(it).toFloat() }
                common.copy(waveform = points.takeLast(800), sampleUnits = root.optString("units", common.sampleUnits))
            }
            "forecast.result" -> common.copy(
                phase = SessionPhase.RESULT,
                acceptedSegments = 20,
                elapsedSeconds = 600,
                forecast = ForecastResult(
                    id = root.optString("forecast_id", "$sessionId-$seq"),
                    score = root.optDouble("score", Double.NaN),
                    threshold = root.optNullableDouble("threshold"),
                    issuedAt = root.optString("timestamp_utc"),
                    observationStart = root.optString("observation_start_utc"),
                    observationEnd = root.optString("observation_end_utc"),
                    horizonMinutes = root.optInt("horizon_minutes", 20),
                    modelId = root.optString("model_id", "unknown"),
                    quality = root.optString("quality").toQuality(),
                ),
                message = "A verified model result is available",
            )
            "forecast.error" -> common.copy(
                phase = SessionPhase.ERROR,
                message = root.optString("message", "Forecast unavailable"),
                forecast = null,
            )
            else -> common
        }
    }

    private fun JSONObject.optNullableDouble(key: String): Double? =
        if (has(key) && !isNull(key)) optDouble(key).takeUnless { it.isNaN() } else null

    private fun String.toMode(fallback: SourceMode) = when (lowercase()) {
        "live" -> SourceMode.LIVE
        "replay" -> SourceMode.REPLAY
        "model_validation", "model_validation_replay" -> SourceMode.MODEL_VALIDATION
        "ui_demo", "fixture" -> SourceMode.UI_DEMO
        else -> fallback
    }
    private fun String.toConnection(fallback: ConnectionState) = when (lowercase()) {
        "connected", "online" -> ConnectionState.CONNECTED
        "connecting" -> ConnectionState.CONNECTING
        "disconnected", "offline" -> ConnectionState.DISCONNECTED
        else -> fallback
    }
    private fun String.toPhase(fallback: SessionPhase) = runCatching { SessionPhase.valueOf(uppercase()) }.getOrDefault(fallback)
    private fun String.toQuality() = when (lowercase()) {
        "valid", "good" -> SignalQuality.GOOD
        "checking" -> SignalQuality.CHECKING
        "low", "poor" -> SignalQuality.LOW
        "gap", "data_gap" -> SignalQuality.DATA_GAP
        else -> SignalQuality.UNAVAILABLE
    }
}
