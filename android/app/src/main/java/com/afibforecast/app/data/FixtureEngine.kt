package com.afibforecast.app.data

import com.afibforecast.app.model.*
import kotlin.math.PI
import kotlin.math.sin
import kotlin.random.Random

object FixtureEngine {
    fun waveform(tick: Long, points: Int = 260): List<Float> = List(points) { i ->
        val x = (i + tick * 4) / 32.0
        val phase = x % 2.4
        val baseline = 0.05 * sin(x * 2 * PI / 3.7)
        val beat = when {
            phase < 0.10 -> -0.10 * sin(phase / 0.10 * PI)
            phase < 0.18 -> 0.95 * sin((phase - 0.10) / 0.08 * PI)
            phase < 0.28 -> -0.28 * sin((phase - 0.18) / 0.10 * PI)
            phase in 0.62..1.05 -> 0.18 * sin((phase - 0.62) / 0.43 * PI)
            else -> 0.0
        }
        (baseline + beat + Random(i + tick.toInt()).nextDouble(-0.012, 0.012)).toFloat()
    }

    fun simulation(tick: Long): MonitorState {
        val elapsed = (tick / 10).toInt().coerceAtMost(600)
        val done = elapsed >= 600
        return MonitorState(
            mode = SourceMode.REPLAY,
            phase = if (done) SessionPhase.RESULT else SessionPhase.COLLECTING,
            sessionId = "simulation-af-07",
            recordingId = "AF-SIM-07",
            acceptedSegments = (elapsed / 30).coerceAtMost(20),
            elapsedSeconds = elapsed,
            signalQuality = SignalQuality.GOOD,
            waveform = waveform(tick),
            forecast = if (done) ForecastResult(
                id = "forecast-sim-07", score = 0.782, threshold = 0.500,
                issuedAt = "2026-11-14T07:10:00Z", observationStart = "2026-11-14T07:00:00Z",
                observationEnd = "2026-11-14T07:10:00Z", modelId = "frozen-v1",
                quality = SignalQuality.GOOD,
            ) else null,
            message = if (done) "UI fixture result shown for presentation only" else "Playing a prerecorded AF example through the observation flow",
            sequence = tick,
            playbackSpeed = "1×",
        )
    }

    fun liveDemo(tick: Long): MonitorState = MonitorState(
        mode = SourceMode.UI_DEMO,
        phase = SessionPhase.COLLECTING,
        piConnection = ConnectionState.DISCONNECTED,
        pcbConnection = ConnectionState.DISCONNECTED,
        sessionId = "live-ui-demo",
        recordingId = null,
        acceptedSegments = 7,
        elapsedSeconds = 214,
        signalQuality = if (tick % 24 in 20..22) SignalQuality.CHECKING else SignalQuality.GOOD,
        waveform = waveform(tick + 80),
        forecast = null,
        message = "UI DEMO — connect the Pi to begin a real live test",
        sequence = tick,
    )
}
