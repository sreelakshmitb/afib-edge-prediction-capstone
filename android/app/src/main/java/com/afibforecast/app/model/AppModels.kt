package com.afibforecast.app.model

enum class SourceMode { LIVE, REPLAY, MODEL_VALIDATION, UI_DEMO }
enum class SessionPhase { READY, COLLECTING, PROCESSING, RESULT, PAUSED, STOPPED, ERROR }
enum class ConnectionState { CONNECTED, CONNECTING, DISCONNECTED }
enum class SignalQuality { GOOD, CHECKING, LOW, DATA_GAP, UNAVAILABLE }

data class ForecastResult(
    val id: String,
    val score: Double,
    val threshold: Double?,
    val issuedAt: String,
    val observationStart: String,
    val observationEnd: String,
    val horizonMinutes: Int = 20,
    val modelId: String,
    val quality: SignalQuality,
    val stale: Boolean = false,
)

data class MonitorState(
    val mode: SourceMode = SourceMode.UI_DEMO,
    val phase: SessionPhase = SessionPhase.COLLECTING,
    val piConnection: ConnectionState = ConnectionState.DISCONNECTED,
    val pcbConnection: ConnectionState = ConnectionState.DISCONNECTED,
    val sessionId: String = "",
    val recordingId: String? = null,
    val acceptedSegments: Int = 0,
    val elapsedSeconds: Int = 0,
    val totalSeconds: Int = 600,
    val signalQuality: SignalQuality = SignalQuality.UNAVAILABLE,
    val lastPacketAt: String = "—",
    val sampleUnits: String = "mV",
    val waveform: List<Float> = emptyList(),
    val forecast: ForecastResult? = null,
    val message: String = "Connect to the Raspberry Pi",
    val sequence: Long = 0,
    val playbackSpeed: String = "1×",
)

data class SessionSummary(
    val id: String,
    val mode: SourceMode,
    val label: String,
    val date: String,
    val duration: String,
    val outcome: String,
    val score: Double? = null,
    val interrupted: Boolean = false,
)

data class ConnectionSettings(
    val baseUrl: String = "http://192.168.4.1:8080",
    val autoConnect: Boolean = true,
    val fixtureMode: Boolean = true,
)

data class AppUiState(
    val monitor: MonitorState = MonitorState(),
    val history: List<SessionSummary> = emptyList(),
    val settings: ConnectionSettings = ConnectionSettings(),
    val selectedExperience: SourceMode = SourceMode.REPLAY,
    val isRefreshing: Boolean = false,
)

val sampleSessions = listOf(
    SessionSummary("AF-SIM-07", SourceMode.REPLAY, "AF simulation", "Today, 11:40 AM", "10 min", "Forecast complete", 0.782),
    SessionSummary("LIVE-BENCH-03", SourceMode.LIVE, "Live bench test", "Today, 10:05 AM", "7 min", "Connection interrupted", interrupted = true),
    SessionSummary("MODEL-VAL-12", SourceMode.MODEL_VALIDATION, "Model validation", "Yesterday, 4:20 PM", "10 min", "Parity verified", 0.108),
)
