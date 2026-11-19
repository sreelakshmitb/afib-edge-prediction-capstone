package com.afibforecast.app

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.afibforecast.app.data.FixtureEngine
import com.afibforecast.app.data.PiClient
import com.afibforecast.app.data.SessionHistoryStore
import com.afibforecast.app.model.*
import kotlinx.coroutines.delay
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainViewModel(application: Application) : AndroidViewModel(application) {
    private val prefs = application.getSharedPreferences("afib_forecast", 0)
    private val historyStore = SessionHistoryStore(prefs)
    private val _state = MutableStateFlow(
        AppUiState(history = historyStore.load(), settings = ConnectionSettings(
            baseUrl = prefs.getString("base_url", "http://192.168.4.1:8080")!!,
            autoConnect = prefs.getBoolean("auto_connect", true),
            fixtureMode = prefs.getBoolean("fixture_mode", true),
        ))
    )
    val state: StateFlow<AppUiState> = _state.asStateFlow()
    private var currentTick = 0L
    private var simulationStartTick = 0L
    private var reconnectJob: Job? = null

    private val piClient = PiClient(
        onEvent = { monitor -> _state.update { withMonitor(it, monitor).copy(isRefreshing = false) } },
        onConnectionChanged = { connected, error ->
            _state.update { old ->
                old.copy(monitor = old.monitor.copy(
                    piConnection = if (connected) ConnectionState.CONNECTED else ConnectionState.DISCONNECTED,
                    forecast = old.monitor.forecast?.copy(stale = !connected),
                    message = error ?: if (connected) "Pi connected — synchronizing session" else old.monitor.message,
                ), isRefreshing = false)
            }
            if (connected) reconnectJob?.cancel()
            if (!connected && error != null && reconnectJob?.isActive != true) {
                reconnectJob = viewModelScope.launch {
                    delay(3_000)
                    if (!_state.value.settings.fixtureMode) connectToPi()
                }
            }
        },
    )

    init {
        startFixtures()
        if (!_state.value.settings.fixtureMode && _state.value.settings.autoConnect) connectToPi()
    }

    fun chooseExperience(mode: SourceMode) {
        _state.update { it.copy(selectedExperience = mode) }
        if (!_state.value.settings.fixtureMode) return
        if (mode != SourceMode.LIVE) simulationStartTick = currentTick
        val current = if (mode == SourceMode.LIVE) FixtureEngine.liveDemo(currentTick) else FixtureEngine.simulation(0)
        _state.update { it.copy(monitor = current) }
    }

    fun updateBaseUrl(value: String) {
        _state.update { it.copy(settings = it.settings.copy(baseUrl = value)) }
        prefs.edit().putString("base_url", value).apply()
    }

    fun setFixtureMode(enabled: Boolean) {
        prefs.edit().putBoolean("fixture_mode", enabled).apply()
        _state.update { it.copy(settings = it.settings.copy(fixtureMode = enabled),
            monitor = MonitorState()) }
        if (enabled) {
            piClient.disconnect()
            chooseExperience(_state.value.selectedExperience)
        } else connectToPi()
    }

    fun connectToPi() {
        _state.update { it.copy(isRefreshing = true, monitor = it.monitor.copy(piConnection = ConnectionState.CONNECTING)) }
        piClient.connect(_state.value.settings.baseUrl, _state.value.monitor)
    }

    fun refresh() {
        if (_state.value.settings.fixtureMode) {
            chooseExperience(_state.value.selectedExperience)
        } else {
            connectToPi()
        }
    }

    fun startReplay(recordingId: String) {
        if (_state.value.settings.fixtureMode) return
        _state.update { it.copy(monitor = it.monitor.copy(
            forecast = null, phase = SessionPhase.COLLECTING,
            recordingId = recordingId, acceptedSegments = 0, elapsedSeconds = 0,
            message = "Waiting for ESP32 to send $recordingId.BIN",
        )) }
        piClient.startReplay(_state.value.settings.baseUrl, recordingId) { error ->
            _state.update { it.copy(monitor = it.monitor.copy(
                phase = SessionPhase.ERROR, message = error, forecast = null,
            )) }
        }
    }

    fun replayFromStart() {
        simulationStartTick = currentTick
        _state.update { it.copy(monitor = FixtureEngine.simulation(0)) }
    }

    private fun startFixtures() = viewModelScope.launch {
        var tick = 0L
        while (isActive) {
            delay(100)
            tick++
            currentTick = tick
            val snapshot = _state.value
            if (!snapshot.settings.fixtureMode) continue
            val monitor = if (snapshot.selectedExperience == SourceMode.LIVE) {
                FixtureEngine.liveDemo(tick)
            } else {
                FixtureEngine.simulation((tick - simulationStartTick).coerceAtLeast(0))
            }
            _state.update { withMonitor(it, monitor) }
        }
    }

    private fun withMonitor(old: AppUiState, monitor: MonitorState): AppUiState {
        val result = monitor.forecast ?: return old.copy(monitor = monitor)
        if (old.history.any { it.id == result.id }) return old.copy(monitor = monitor)
        val entry = SessionSummary(
            id = result.id,
            mode = monitor.mode,
            label = when (monitor.mode) {
                SourceMode.LIVE -> "Live test"
                SourceMode.REPLAY -> "AF simulation"
                SourceMode.MODEL_VALIDATION -> "Model validation"
                SourceMode.UI_DEMO -> "UI demonstration"
            },
            date = SimpleDateFormat("dd MMM yyyy, h:mm a", Locale.getDefault()).format(Date()),
            duration = "10 min",
            outcome = "Model score recorded",
            score = result.score,
        )
        val history = (listOf(entry) + old.history).take(100)
        historyStore.save(history)
        return old.copy(monitor = monitor, history = history)
    }

    override fun onCleared() {
        piClient.disconnect()
        super.onCleared()
    }
}
