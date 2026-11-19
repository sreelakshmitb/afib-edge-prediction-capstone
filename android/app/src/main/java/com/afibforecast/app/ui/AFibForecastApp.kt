package com.afibforecast.app.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.afibforecast.app.model.*
import com.afibforecast.app.ui.theme.*
import java.util.Locale

private enum class Page { MONITOR, HISTORY, ABOUT }

private fun ForecastResult.isElevated(): Boolean =
    score.isFinite() && threshold?.takeIf { it.isFinite() }?.let { score >= it } == true

@Composable
fun AFibForecastApp(
    state: AppUiState,
    onExperienceChange: (SourceMode) -> Unit,
    onBaseUrlChange: (String) -> Unit,
    onFixtureChange: (Boolean) -> Unit,
    onReplay: () -> Unit,
    onReplayStart: (String) -> Unit,
    onRefresh: () -> Unit,
) {
    var page by remember { mutableStateOf(Page.MONITOR) }
    var showConnection by remember { mutableStateOf(false) }
    var showHelp by remember { mutableStateOf(false) }
    var showSettings by remember { mutableStateOf(false) }
    var activeRunMode by remember { mutableStateOf<SourceMode?>(null) }
    var ignoredForecastId by remember { mutableStateOf<String?>(null) }
    var waitingForRunReset by remember { mutableStateOf(false) }
    var selectedRecording by remember { mutableStateOf("POS102") }
    var closeAfterConnection by remember { mutableStateOf(false) }
    var liveTestOpened by remember { mutableStateOf(false) }

    LaunchedEffect(showConnection, state.monitor.piConnection, state.settings.fixtureMode, closeAfterConnection) {
        if (showConnection && closeAfterConnection && !state.settings.fixtureMode &&
            state.monitor.piConnection == ConnectionState.CONNECTED) {
            showConnection = false
            closeAfterConnection = false
            page = Page.MONITOR
        }
    }

    LaunchedEffect(activeRunMode, state.monitor.forecast) {
        if (activeRunMode != null && waitingForRunReset && state.monitor.forecast == null) waitingForRunReset = false
    }

    if (activeRunMode != null) {
        Box(Modifier.fillMaxSize().background(Cream).safeDrawingPadding()) {
            AmbientBackground()
            if (!waitingForRunReset && state.monitor.forecast != null && state.monitor.forecast.id != ignoredForecastId) {
                CompletedTestScreen(
                    monitor = state.monitor,
                    onDone = { activeRunMode = null },
                    onAgain = {
                        val mode = activeRunMode!!
                        if (mode == SourceMode.LIVE || !state.settings.fixtureMode) {
                            activeRunMode = null
                        } else {
                            ignoredForecastId = null
                            waitingForRunReset = true
                            onExperienceChange(mode)
                            onReplay()
                        }
                    },
                )
            } else {
                RunningTestScreen(
                    monitor = state.monitor,
                    requestedMode = activeRunMode!!,
                    isFixture = state.settings.fixtureMode,
                    onBack = { activeRunMode = null },
                )
            }
        }
    } else Scaffold(
            containerColor = Cream,
            contentWindowInsets = WindowInsets.safeDrawing,
            topBar = {
                AppHeader(
                    connection = if (state.settings.fixtureMode) ConnectionState.DISCONNECTED else state.monitor.piConnection,
                    refreshing = state.isRefreshing,
                    onRefresh = onRefresh,
                    onHelp = { showHelp = true },
                    onSettings = { showSettings = true },
                    onConnection = { showConnection = true },
                )
            },
            bottomBar = { BottomNavigation(page) { page = it } },
        ) { padding ->
            Box(Modifier.fillMaxSize().padding(padding)) {
                AmbientBackground()
                when (page) {
                    Page.MONITOR -> HomeScreen(
                        state = state,
                        selectedRecording = selectedRecording,
                        liveTestOpened = liveTestOpened,
                        onRecordingChange = { selectedRecording = it },
                        onExperienceChange = onExperienceChange,
                        onConnect = { showConnection = true },
                        onStart = { mode ->
                            if (mode == SourceMode.LIVE) liveTestOpened = true
                            ignoredForecastId = if (mode == SourceMode.LIVE || state.settings.fixtureMode) null else state.monitor.forecast?.id
                            waitingForRunReset = mode != SourceMode.LIVE && state.settings.fixtureMode
                            onExperienceChange(mode)
                            if (mode != SourceMode.LIVE && state.settings.fixtureMode) onReplay()
                            if (mode == SourceMode.REPLAY && !state.settings.fixtureMode) onReplayStart(selectedRecording)
                            activeRunMode = mode
                        },
                    )
                    Page.HISTORY -> HistoryScreen(state.history)
                    Page.ABOUT -> AboutScreen()
                }
            }
        }
    if (showConnection) ConnectionDialog(
        settings = state.settings,
        isConnecting = state.isRefreshing,
        onDismiss = { showConnection = false },
        onUrlChange = onBaseUrlChange,
        onFixtureChange = { enabled ->
            closeAfterConnection = !enabled
            onFixtureChange(enabled)
        },
    )
    if (showHelp) HelpDialog(onDismiss = { showHelp = false })
    if (showSettings) EmergencyContactSettings(onDismiss = { showSettings = false })
}

@Composable
private fun AmbientBackground() {
    Canvas(Modifier.fillMaxSize()) {
        drawCircle(SagePale.copy(alpha = .56f), size.width * .50f, Offset(size.width * .92f, size.height * .08f))
        drawCircle(AmberSoft.copy(alpha = .18f), size.width * .42f, Offset(size.width * .08f, size.height * .54f))
    }
}

@Composable
private fun AppHeader(
    connection: ConnectionState,
    refreshing: Boolean,
    onRefresh: () -> Unit,
    onHelp: () -> Unit,
    onSettings: () -> Unit,
    onConnection: () -> Unit,
) {
    var refreshClicks by remember { mutableIntStateOf(0) }
    val refreshAngle by animateFloatAsState(
        targetValue = refreshClicks * 360f,
        animationSpec = tween(700, easing = FastOutSlowInEasing),
        label = "Refresh rotation",
    )
    Row(
        Modifier.fillMaxWidth().background(Cream.copy(alpha = .96f)).statusBarsPadding()
            .padding(horizontal = 22.dp).padding(top = 12.dp, bottom = 14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically) {
            BrandMark(Modifier.size(50.dp))
            Spacer(Modifier.width(11.dp))
            Column {
                Text("AFib", style = MaterialTheme.typography.titleLarge, color = Brown)
                Text("FORECAST", style = MaterialTheme.typography.labelMedium, color = OliveDeep, letterSpacing = 1.6.sp)
            }
        }
        Surface(
            modifier = Modifier.size(44.dp).clickable(role = Role.Button, onClick = onSettings)
                .semantics { contentDescription = "Emergency contacts" },
            shape = CircleShape, color = Paper, shadowElevation = 2.dp,
        ) {
            Box(contentAlignment = Alignment.Center) {
                AppGlyph(Glyph.EMERGENCY_CONTACT, ErrorRed, Modifier.size(25.dp))
            }
        }
        Spacer(Modifier.width(8.dp))
        Surface(
            modifier = Modifier.size(44.dp).clickable(role = Role.Button, onClick = onHelp)
                .semantics { contentDescription = "How to use the app" },
            shape = CircleShape, color = Paper, shadowElevation = 2.dp,
        ) {
            Box(contentAlignment = Alignment.Center) {
                Text("?", style = MaterialTheme.typography.titleLarge, color = Brown)
            }
        }
        Spacer(Modifier.width(8.dp))
        Surface(
            modifier = Modifier.size(44.dp).clickable(role = Role.Button) {
                refreshClicks++
                onRefresh()
            }
                .semantics { contentDescription = "Refresh Pi connection" },
            shape = CircleShape, color = Paper, shadowElevation = 2.dp,
        ) {
            Box(contentAlignment = Alignment.Center) {
                AppGlyph(Glyph.REFRESH, if (refreshing) Orange else Brown,
                    Modifier.size(23.dp).graphicsLayer { rotationZ = refreshAngle })
            }
        }
        Spacer(Modifier.width(8.dp))
        Surface(
            modifier = Modifier.size(52.dp).clickable(role = Role.Button, onClick = onConnection)
                .semantics { contentDescription = "Open connection settings" },
            shape = CircleShape,
            color = Paper,
            shadowElevation = 2.dp,
        ) {
            Box(contentAlignment = Alignment.Center) {
                AppGlyph(Glyph.CONNECTION, if (connection == ConnectionState.CONNECTED) OliveDeep else BrownMuted, Modifier.size(25.dp))
                Box(Modifier.align(Alignment.TopEnd).offset((-5).dp, 5.dp).size(11.dp).clip(CircleShape).background(
                    if (connection == ConnectionState.CONNECTED) Olive else Orange
                ))
            }
        }
    }
}

@Composable
private fun BrandMark(modifier: Modifier = Modifier) {
    Canvas(modifier.clip(RoundedCornerShape(16.dp)).background(SagePale)) {
        drawCircle(Paper, size.minDimension * .37f, center)
        val pulse = Path().apply {
            moveTo(size.width * .17f, size.height * .53f)
            lineTo(size.width * .34f, size.height * .53f)
            lineTo(size.width * .43f, size.height * .32f)
            lineTo(size.width * .55f, size.height * .72f)
            lineTo(size.width * .66f, size.height * .44f)
            lineTo(size.width * .75f, size.height * .53f)
            lineTo(size.width * .85f, size.height * .53f)
        }
        drawPath(pulse, ActionBrown, style = Stroke(width = size.width * .065f, cap = StrokeCap.Round, join = androidx.compose.ui.graphics.StrokeJoin.Round))
    }
}

@Composable
private fun BottomNavigation(selected: Page, onSelect: (Page) -> Unit) {
    Surface(color = Paper, shadowElevation = 16.dp) {
        Row(
            Modifier.fillMaxWidth().navigationBarsPadding().height(78.dp).padding(horizontal = 16.dp),
            horizontalArrangement = Arrangement.SpaceAround,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            NavItem(Page.MONITOR, "Monitor", Glyph.WAVE, selected, onSelect)
            NavItem(Page.HISTORY, "Sessions", Glyph.HISTORY, selected, onSelect)
            NavItem(Page.ABOUT, "About", Glyph.INFO, selected, onSelect)
        }
    }
}

@Composable
private fun RowScope.NavItem(page: Page, label: String, glyph: Glyph, selected: Page, onSelect: (Page) -> Unit) {
    val active = selected == page
    Column(
        Modifier.weight(1f).heightIn(min = 56.dp).clip(RoundedCornerShape(22.dp))
            .background(if (active) SagePale else Color.Transparent)
            .clickable(role = Role.Tab) { onSelect(page) }.padding(vertical = 8.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        AppGlyph(glyph, if (active) OliveDeep else BrownMuted, Modifier.size(23.dp))
        Spacer(Modifier.height(3.dp))
        Text(label, style = MaterialTheme.typography.labelMedium, color = if (active) Brown else BrownMuted)
    }
}

@Composable
private fun HomeScreen(
    state: AppUiState,
    selectedRecording: String,
    liveTestOpened: Boolean,
    onRecordingChange: (String) -> Unit,
    onExperienceChange: (SourceMode) -> Unit,
    onConnect: () -> Unit,
    onStart: (SourceMode) -> Unit,
) {
    val selectedMode = state.selectedExperience
    val piConnected = !state.settings.fixtureMode &&
        state.monitor.piConnection == ConnectionState.CONNECTED
    val equipmentReady = state.monitor.piConnection == ConnectionState.CONNECTED &&
        state.monitor.pcbConnection == ConnectionState.CONNECTED && !state.settings.fixtureMode
    val detectedMode = when {
        state.monitor.mode == SourceMode.LIVE -> SourceMode.LIVE
        state.monitor.mode == SourceMode.REPLAY || state.monitor.mode == SourceMode.MODEL_VALIDATION -> SourceMode.REPLAY
        else -> null
    }
    val liveInProgress = !state.settings.fixtureMode && selectedMode == SourceMode.LIVE &&
        state.monitor.mode == SourceMode.LIVE &&
        (liveTestOpened || state.monitor.phase in listOf(SessionPhase.COLLECTING, SessionPhase.PROCESSING)) &&
        state.monitor.forecast == null
    val liveResultAvailable = !state.settings.fixtureMode && selectedMode == SourceMode.LIVE &&
        state.monitor.mode == SourceMode.LIVE && state.monitor.forecast != null
    val canStart = if (!state.settings.fixtureMode && selectedMode == SourceMode.REPLAY) {
        state.monitor.piConnection == ConnectionState.CONNECTED && detectedMode == SourceMode.REPLAY &&
            state.monitor.phase != SessionPhase.COLLECTING && state.monitor.phase != SessionPhase.PROCESSING
    } else if (selectedMode == SourceMode.LIVE) {
        (equipmentReady && detectedMode == SourceMode.LIVE) || liveInProgress || liveResultAvailable
    } else equipmentReady && detectedMode == selectedMode

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp).padding(bottom = 28.dp),
    ) {
        Text("Heart monitoring.\nMade simple.", style = MaterialTheme.typography.displayMedium, color = Brown)
        Spacer(Modifier.height(8.dp))
        Text("A calm, guided check when you need it.", style = MaterialTheme.typography.bodyLarge, color = BrownMuted)
        Spacer(Modifier.height(22.dp))

        ModeSections(selectedMode, detectedMode, onExperienceChange)
        if (selectedMode == SourceMode.REPLAY && !state.settings.fixtureMode) {
            Spacer(Modifier.height(10.dp))
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text("Pre-recorded ECG", style = MaterialTheme.typography.labelMedium,
                    color = BrownMuted, maxLines = 1, modifier = Modifier.weight(1f))
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    listOf("POS102", "NEG024").forEach { name ->
                        FilterChip(selected = selectedRecording == name,
                            onClick = { onRecordingChange(name) }, label = { Text(name) })
                    }
                }
            }
        }
        Spacer(Modifier.height(14.dp))
        ReadinessCard(
            monitor = state.monitor,
            selectedMode = selectedMode,
            detectedMode = detectedMode,
            fixtureMode = state.settings.fixtureMode,
            ready = if (selectedMode == SourceMode.REPLAY && !state.settings.fixtureMode)
                state.monitor.piConnection == ConnectionState.CONNECTED else equipmentReady,
        )
        Spacer(Modifier.height(14.dp))

        OutlinedButton(
            onClick = onConnect,
            modifier = Modifier.fillMaxWidth().heightIn(min = 60.dp),
            shape = RoundedCornerShape(21.dp),
            border = BorderStroke(1.5.dp, if (piConnected) OliveDeep else ActionBrown),
            colors = ButtonDefaults.outlinedButtonColors(
                containerColor = if (piConnected) Olive else ActionBrown,
                contentColor = Paper,
            ),
        ) {
            AppGlyph(if (piConnected) Glyph.CHECK else Glyph.CONNECTION, Paper, Modifier.size(24.dp))
            Spacer(Modifier.width(11.dp))
            Text(when {
                piConnected -> "Connection successful"
                state.isRefreshing -> "Connecting…"
                else -> "Connect equipment"
            }, style = MaterialTheme.typography.titleMedium)
        }
        Spacer(Modifier.height(10.dp))

        Button(
            onClick = { onStart(selectedMode) },
            enabled = canStart,
            modifier = Modifier.fillMaxWidth().heightIn(min = 68.dp),
            shape = RoundedCornerShape(23.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = ActionBrown,
                contentColor = Paper,
                disabledContainerColor = Brown.copy(alpha = .28f),
                disabledContentColor = Paper,
            ),
        ) {
            AppGlyph(Glyph.PLAY, Paper, Modifier.size(26.dp))
            Spacer(Modifier.width(12.dp))
            Text(
                when {
                    liveResultAvailable -> "View live result"
                    liveInProgress -> "View live status"
                    selectedMode == SourceMode.LIVE -> "Start live testing"
                    selectedMode == SourceMode.REPLAY -> "Start AF simulation"
                    else -> "Start testing"
                },
                style = MaterialTheme.typography.titleMedium,
            )
        }
    }
}

@Composable
private fun ModeSections(selectedMode: SourceMode, detectedMode: SourceMode?, onSelect: (SourceMode) -> Unit) {
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(24.dp)).background(Paper).padding(5.dp),
    ) {
        ModeSection("Live testing", selectedMode == SourceMode.LIVE, detectedMode == SourceMode.LIVE, Modifier.weight(1f)) { onSelect(SourceMode.LIVE) }
        ModeSection("AF simulation", selectedMode != SourceMode.LIVE, detectedMode == SourceMode.REPLAY, Modifier.weight(1f)) { onSelect(SourceMode.REPLAY) }
    }
}

@Composable
private fun ModeSection(label: String, selected: Boolean, physicalActive: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val backgroundColor by animateColorAsState(
        if (selected) ActionBrown else Color.Transparent,
        animationSpec = tween(320, easing = FastOutSlowInEasing),
        label = "mode section",
    )
    val textColor by animateColorAsState(
        if (selected) Paper else BrownMuted,
        animationSpec = tween(320), label = "mode text",
    )
    Row(
        modifier.heightIn(min = 56.dp).clip(RoundedCornerShape(19.dp))
            .background(backgroundColor)
            .clickable(role = Role.Tab, onClick = onClick).padding(horizontal = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.Center,
    ) {
        PulsingStatusDot(physicalActive, 11)
        Spacer(Modifier.width(8.dp))
        Text(label, style = MaterialTheme.typography.labelLarge, color = textColor)
    }
}

@Composable
private fun ReadinessCard(
    monitor: MonitorState,
    selectedMode: SourceMode,
    detectedMode: SourceMode?,
    fixtureMode: Boolean,
    ready: Boolean,
) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(30.dp), color = Paper, shadowElevation = 2.dp) {
        Column(Modifier.padding(20.dp)) {
            Text(
                when {
                    !ready -> "Connect your equipment"
                    ready && detectedMode != selectedMode -> "Selected mode is not ready"
                    ready && detectedMode == SourceMode.LIVE -> "Ready for live testing"
                    ready && detectedMode == SourceMode.REPLAY -> "Ready for AF simulation"
                    else -> "Connect your equipment"
                },
                style = MaterialTheme.typography.headlineMedium, color = Brown, maxLines = 1,
            )
            Spacer(Modifier.height(6.dp))
            Text(
                when {
                    fixtureMode && selectedMode == SourceMode.LIVE -> "UI demo — connect Pi and ECG."
                    fixtureMode -> "UI demo — connect Pi for replay."
                    !ready && selectedMode == SourceMode.LIVE -> "Connect Pi and ECG device."
                    !ready -> "Connect Pi for replay."
                    detectedMode != selectedMode -> "Select the Pi's current mode."
                    selectedMode == SourceMode.LIVE -> "Collect 10 minutes of live ECG."
                    else -> "Replay a saved 10-minute ECG recording."
                },
                style = MaterialTheme.typography.bodyMedium, color = BrownMuted, maxLines = 1, overflow = TextOverflow.Ellipsis,
            )
            Spacer(Modifier.height(18.dp))
            ConnectionLine("Raspberry Pi", if (fixtureMode) ConnectionState.DISCONNECTED else monitor.piConnection)
            if (selectedMode == SourceMode.LIVE) {
                Spacer(Modifier.height(10.dp))
                ConnectionLine("ECG module", if (fixtureMode) ConnectionState.DISCONNECTED else monitor.pcbConnection)
                Spacer(Modifier.height(10.dp))
                ModeConnectionLine("Signal source", "Live ECG", ready && detectedMode == SourceMode.LIVE)
            } else {
                Spacer(Modifier.height(10.dp))
                ModeConnectionLine("Signal source", "Saved ECG recording", ready && detectedMode == SourceMode.REPLAY)
            }
        }
    }
}

@Composable
private fun ConnectionLine(label: String, state: ConnectionState) {
    val connected = state == ConnectionState.CONNECTED
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        PulsingStatusDot(connected, 13)
        Spacer(Modifier.width(11.dp))
        Text(label, style = MaterialTheme.typography.bodyLarge, color = Brown, modifier = Modifier.weight(1f))
        Text(
            when (state) { ConnectionState.CONNECTED -> "Connected"; ConnectionState.CONNECTING -> "Connecting"; ConnectionState.DISCONNECTED -> "Not connected" },
            style = MaterialTheme.typography.labelMedium,
            color = if (connected) OliveDeep else BrownMuted,
        )
    }
}

@Composable
private fun ModeConnectionLine(label: String, activeLabel: String, connected: Boolean) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        PulsingStatusDot(connected, 13)
        Spacer(Modifier.width(11.dp))
        Text(label, style = MaterialTheme.typography.bodyLarge, color = Brown, modifier = Modifier.weight(1f))
        Text(
            if (connected) activeLabel else "Not ready",
            style = MaterialTheme.typography.labelMedium,
            color = if (connected) OliveDeep else BrownMuted,
        )
    }
}

@Composable
private fun PulsingStatusDot(active: Boolean, sizeDp: Int) {
    val transition = rememberInfiniteTransition(label = "Ready status pulse")
    val scale by transition.animateFloat(
        initialValue = 0.78f, targetValue = 1.18f,
        animationSpec = infiniteRepeatable(tween(950, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "Ready dot scale",
    )
    Box(Modifier.size(sizeDp.dp).graphicsLayer {
        scaleX = if (active) scale else 1f
        scaleY = if (active) scale else 1f
    }.clip(CircleShape).background(if (active) Olive else Line))
}

@Composable
private fun ModeDisclosure(monitor: MonitorState, fixture: Boolean) {
    val isLive = monitor.mode == SourceMode.LIVE && !fixture
    val bg = when { isLive -> SagePale; monitor.mode == SourceMode.REPLAY -> AmberSoft.copy(alpha = .55f); else -> Lavender.copy(alpha = .22f) }
    val title = when {
        isLive -> "LIVE TEST"
        fixture && monitor.mode == SourceMode.REPLAY -> "AF SIMULATION • UI DEMO"
        monitor.mode == SourceMode.REPLAY -> "PRERECORDED AF SIMULATION"
        monitor.mode == SourceMode.MODEL_VALIDATION -> "MODEL VALIDATION REPLAY"
        else -> "UI DEMO — NOT LIVE DATA"
    }
    val detail = when {
        isLive -> "The Pi is the source of truth for this session."
        fixture && monitor.mode == SourceMode.REPLAY -> "Fixture example ${monitor.recordingId ?: "recording"} • ${monitor.playbackSpeed} • not a patient or Pi result"
        monitor.mode == SourceMode.REPLAY -> "${monitor.recordingId ?: "recording"} • accelerated transfer of a 10-minute ECG recording • not a patient signal"
        monitor.mode == SourceMode.MODEL_VALIDATION -> "Already preprocessed input used to verify model parity."
        else -> "Connect the Raspberry Pi before conducting a live test."
    }
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(22.dp)).background(bg).padding(16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(Modifier.size(46.dp).clip(CircleShape).background(Paper), contentAlignment = Alignment.Center) {
            AppGlyph(if (isLive) Glyph.HEART else Glyph.PLAY, Brown, Modifier.size(23.dp))
        }
        Spacer(Modifier.width(13.dp))
        Column {
            Text(title, style = MaterialTheme.typography.labelMedium, color = Brown, letterSpacing = .7.sp)
            Spacer(Modifier.height(3.dp))
            Text(detail, style = MaterialTheme.typography.bodyMedium, color = BrownMuted)
        }
    }
}

@Composable
private fun EcgCard(monitor: MonitorState) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(30.dp), color = Brown, shadowElevation = 3.dp) {
        Column(Modifier.padding(20.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("ECG preview", style = MaterialTheme.typography.titleLarge, color = Paper)
                    Text("Recent 8 seconds • ${monitor.sampleUnits}", style = MaterialTheme.typography.bodyMedium, color = Paper.copy(alpha = .68f))
                }
                QualityPill(monitor.signalQuality)
            }
            Spacer(Modifier.height(12.dp))
            EcgWaveform(monitor.waveform, Modifier.fillMaxWidth().height(142.dp))
            Spacer(Modifier.height(8.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("8 sec ago", style = MaterialTheme.typography.labelMedium, color = Paper.copy(alpha = .55f))
                Text("Now", style = MaterialTheme.typography.labelMedium, color = Paper.copy(alpha = .8f))
            }
        }
    }
}

@Composable
private fun QualityPill(quality: SignalQuality) {
    val label = when (quality) { SignalQuality.GOOD -> "Signal good"; SignalQuality.CHECKING -> "Checking"; SignalQuality.LOW -> "Low signal"; SignalQuality.DATA_GAP -> "Data gap"; else -> "Unavailable" }
    val color = if (quality == SignalQuality.GOOD) Olive else Orange
    Row(Modifier.clip(RoundedCornerShape(50)).background(Paper.copy(alpha = .12f)).padding(horizontal = 11.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(9.dp).clip(CircleShape).background(color)); Spacer(Modifier.width(7.dp))
        Text(label, style = MaterialTheme.typography.labelMedium, color = Paper)
    }
}

@Composable
private fun EcgWaveform(values: List<Float>, modifier: Modifier) {
    Canvas(modifier.semantics { contentDescription = "ECG waveform preview; this is not a forecast result" }) {
        repeat(5) { row -> drawLine(Paper.copy(alpha = .07f), Offset(0f, size.height * row / 4), Offset(size.width, size.height * row / 4), 1f) }
        repeat(9) { col -> drawLine(Paper.copy(alpha = .05f), Offset(size.width * col / 8, 0f), Offset(size.width * col / 8, size.height), 1f) }
        if (values.size > 1) {
            val min = values.minOrNull() ?: -1f; val max = values.maxOrNull() ?: 1f; val range = (max - min).coerceAtLeast(.1f)
            val p = Path()
            values.forEachIndexed { index, value ->
                val x = index.toFloat() / (values.lastIndex) * size.width
                val y = size.height - ((value - min) / range * size.height * .74f + size.height * .13f)
                if (index == 0) p.moveTo(x, y) else p.lineTo(x, y)
            }
            drawPath(p, AmberSoft, style = Stroke(width = 4f, cap = StrokeCap.Round))
        }
    }
}

@Composable
private fun ObservationCard(monitor: MonitorState) {
    val progress = monitor.acceptedSegments / 20f
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(30.dp), color = Paper, shadowElevation = 2.dp) {
        Column(Modifier.padding(20.dp)) {
            Row(verticalAlignment = Alignment.Bottom) {
                Column(Modifier.weight(1f)) {
                    Text("Observation", style = MaterialTheme.typography.titleLarge, color = Brown)
                    Text("A full 10 minutes is needed", style = MaterialTheme.typography.bodyMedium, color = BrownMuted)
                }
                Text(formatTime(monitor.elapsedSeconds), style = MaterialTheme.typography.headlineMedium, color = Brown)
            }
            Spacer(Modifier.height(20.dp))
            Canvas(Modifier.fillMaxWidth().height(22.dp).semantics { contentDescription = "${monitor.acceptedSegments} of 20 valid segments collected" }) {
                drawRoundRect(Line, cornerRadius = androidx.compose.ui.geometry.CornerRadius(size.height / 2))
                drawRoundRect(Olive, size = androidx.compose.ui.geometry.Size(size.width * progress, size.height), cornerRadius = androidx.compose.ui.geometry.CornerRadius(size.height / 2))
            }
            Spacer(Modifier.height(12.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("${monitor.acceptedSegments} of 20 valid segments", style = MaterialTheme.typography.labelLarge, color = OliveDeep)
                Text("10:00", style = MaterialTheme.typography.labelLarge, color = BrownMuted)
            }
            Spacer(Modifier.height(12.dp))
            Text(monitor.message, style = MaterialTheme.typography.bodyMedium, color = BrownMuted)
        }
    }
}

@Composable
private fun ForecastCard(monitor: MonitorState) {
    val result = monitor.forecast
    val waiting = result == null
    val elevated = result?.isElevated() == true
    Surface(
        Modifier.fillMaxWidth(), shape = RoundedCornerShape(30.dp),
        color = when { waiting -> SagePale; elevated -> RiskPale; else -> Olive }, shadowElevation = 2.dp,
    ) {
        Column(Modifier.padding(22.dp)) {
            Text(when { waiting -> "Forecast"; elevated -> "Elevated forecast risk"; else -> "Model result" },
                style = MaterialTheme.typography.titleLarge, color = if (elevated) ErrorRed else Brown)
            Spacer(Modifier.height(10.dp))
            if (waiting) {
                Text("Waiting for a complete observation", style = MaterialTheme.typography.headlineMedium, color = Brown)
                Spacer(Modifier.height(8.dp))
                Text("No risk percentage is shown during warm-up. The waveform above is only a signal preview.", style = MaterialTheme.typography.bodyLarge, color = BrownMuted)
            } else {
                if (result.stale) {
                    Text("Pi disconnected — this result is from an earlier replay session.",
                        style = MaterialTheme.typography.bodyLarge, color = ErrorRed)
                    Spacer(Modifier.height(8.dp))
                }
                if (monitor.mode == SourceMode.REPLAY) {
                    Text("PRERECORDED REPLAY • ${monitor.recordingId ?: "unknown recording"}",
                        style = MaterialTheme.typography.labelLarge, color = Brown)
                    Spacer(Modifier.height(8.dp))
                }
                Row(verticalAlignment = Alignment.Bottom) {
                    Text(String.format(Locale.US, "%.3f", result.score), style = MaterialTheme.typography.displayLarge,
                        color = if (elevated) ErrorRed else Brown)
                    Spacer(Modifier.width(10.dp)); Text("model score", style = MaterialTheme.typography.bodyLarge, color = BrownMuted, modifier = Modifier.padding(bottom = 7.dp))
                }
                Spacer(Modifier.height(7.dp))
                Text(
                    if (result.threshold == null) "Interpretation unavailable — no validated threshold is configured."
                    else if (elevated) "The score is above the configured research threshold ${String.format(Locale.US, "%.3f", result.threshold)}."
                    else "Compared with the validated operating threshold ${String.format(Locale.US, "%.3f", result.threshold)}.",
                    style = MaterialTheme.typography.bodyLarge, color = if (elevated) ErrorRed else Brown,
                )
                HorizontalDivider(Modifier.padding(vertical = 16.dp), color = Brown.copy(alpha = .18f))
                DetailLine("Forecast horizon", "${result.horizonMinutes} minutes")
                DetailLine("Model", result.modelId)
                DetailLine("Observation", "10 complete minutes")
            }
            Spacer(Modifier.height(14.dp))
            Text("Research prototype • not a diagnosis or emergency alert", style = MaterialTheme.typography.labelMedium, color = Brown)
        }
    }
}

@Composable
private fun RunningTestScreen(
    monitor: MonitorState,
    requestedMode: SourceMode,
    isFixture: Boolean,
    onBack: () -> Unit,
) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp).padding(bottom = 28.dp)) {
        Row(Modifier.fillMaxWidth().padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
            Surface(
                Modifier.size(52.dp).clickable(role = Role.Button, onClick = onBack),
                shape = CircleShape, color = Paper, shadowElevation = 2.dp,
            ) { Box(contentAlignment = Alignment.Center) { AppGlyph(Glyph.BACK, Brown, Modifier.size(24.dp)) } }
            Spacer(Modifier.width(14.dp))
            Column(Modifier.weight(1f)) {
                Text(if (requestedMode == SourceMode.LIVE) "Live test" else "AF simulation", style = MaterialTheme.typography.titleLarge, color = Brown)
                Text(when {
                    isFixture -> "UI DEMONSTRATION"
                    monitor.piConnection != ConnectionState.CONNECTED -> "RECONNECTING TO PI"
                    monitor.phase == SessionPhase.PAUSED -> "WAITING FOR LIVE UPDATES"
                    else -> "TEST IN PROGRESS"
                }, style = MaterialTheme.typography.labelMedium, color = OliveDeep, letterSpacing = .8.sp)
            }
        }
        Spacer(Modifier.height(18.dp))
        Text(if (requestedMode == SourceMode.REPLAY) "Receiving prerecorded ECG." else "Stay comfortable.\nBreathe normally.", style = MaterialTheme.typography.headlineLarge, color = Brown)
        Spacer(Modifier.height(8.dp))
        Text(if (requestedMode == SourceMode.REPLAY) "The Pi verifies the complete 10-minute recording before scoring." else "Keep still while the observation is collected.", style = MaterialTheme.typography.bodyLarge, color = BrownMuted)
        Spacer(Modifier.height(14.dp))
        RunningConnectionStrip(monitor, requestedMode, isFixture)
        Spacer(Modifier.height(18.dp))
        ImmersiveTimer(monitor.elapsedSeconds, monitor.totalSeconds, requestedMode == SourceMode.REPLAY)
        Spacer(Modifier.height(18.dp))
        SegmentProgress(monitor.acceptedSegments, requestedMode == SourceMode.REPLAY)
        Spacer(Modifier.height(14.dp))
        Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(24.dp), color = SagePale) {
            Text(
                if (isFixture) "This is a presentation fixture, not a live patient recording or Pi result."
                else monitor.message,
                Modifier.padding(18.dp), style = MaterialTheme.typography.bodyLarge, color = Brown,
            )
        }
    }
}

@Composable
private fun RunningConnectionStrip(monitor: MonitorState, mode: SourceMode, fixture: Boolean) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(22.dp), color = Paper) {
        Row(Modifier.padding(horizontal = 14.dp, vertical = 13.dp), horizontalArrangement = Arrangement.SpaceBetween) {
            MiniStatus("Pi", !fixture && monitor.piConnection == ConnectionState.CONNECTED)
            if (mode == SourceMode.LIVE) {
                MiniStatus("ECG module", !fixture && monitor.pcbConnection == ConnectionState.CONNECTED)
                MiniStatus("Live signal", !fixture && monitor.signalQuality == SignalQuality.GOOD)
            } else {
                MiniStatus("Replay device", !fixture && monitor.pcbConnection == ConnectionState.CONNECTED)
            }
        }
    }
}

@Composable
private fun RowScope.MiniStatus(label: String, ready: Boolean) {
    Row(Modifier.weight(1f), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.Center) {
        Box(Modifier.size(10.dp).clip(CircleShape).background(if (ready) Olive else Line))
        Spacer(Modifier.width(7.dp))
        Text(label, style = MaterialTheme.typography.labelMedium, color = Brown, maxLines = 1)
    }
}

@Composable
private fun ImmersiveTimer(elapsedSeconds: Int, totalSeconds: Int, replay: Boolean) {
    val progress by animateFloatAsState(
        targetValue = (elapsedSeconds.toFloat() / totalSeconds.coerceAtLeast(1)).coerceIn(0f, 1f),
        animationSpec = tween(700, easing = FastOutSlowInEasing), label = "observation progress",
    )
    val breathing = rememberInfiniteTransition(label = "calm breathing")
    val breath by breathing.animateFloat(
        initialValue = .92f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(3200, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "breathing ring",
    )
    Box(Modifier.fillMaxWidth().height(310.dp), contentAlignment = Alignment.Center) {
        Canvas(Modifier.size(292.dp).semantics { contentDescription = "${formatTime(elapsedSeconds)} of ${formatTime(totalSeconds)} elapsed" }) {
            val center = Offset(size.width / 2, size.height / 2)
            drawCircle(SagePale.copy(alpha = .42f), radius = size.minDimension * .49f * breath, center = center)
            drawCircle(SagePale.copy(alpha = .75f), radius = size.minDimension * .40f * breath, center = center)
            drawCircle(Paper, radius = size.minDimension * .31f, center = center)
            val inset = size.minDimension * .065f
            drawArc(
                color = Line, startAngle = -90f, sweepAngle = 360f, useCenter = false,
                topLeft = Offset(inset, inset), size = androidx.compose.ui.geometry.Size(size.width - inset * 2, size.height - inset * 2),
                style = Stroke(width = 13f, cap = StrokeCap.Round),
            )
            drawArc(
                color = OliveDeep, startAngle = -90f, sweepAngle = progress * 360f, useCenter = false,
                topLeft = Offset(inset, inset), size = androidx.compose.ui.geometry.Size(size.width - inset * 2, size.height - inset * 2),
                style = Stroke(width = 13f, cap = StrokeCap.Round),
            )
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(formatTime(elapsedSeconds), style = MaterialTheme.typography.displayLarge, color = Brown)
            Text("of ${formatTime(totalSeconds)}", style = MaterialTheme.typography.bodyLarge, color = BrownMuted)
            Spacer(Modifier.height(7.dp))
            Text(if (replay) "RECORDED ECG DURATION" else "OBSERVING",
                style = MaterialTheme.typography.labelMedium, color = OliveDeep, letterSpacing = 1.5.sp)
        }
    }
    if (replay) Text("Accelerated replay: this is ECG duration received, not time spent recording now.",
        style = MaterialTheme.typography.bodyMedium, color = BrownMuted, textAlign = TextAlign.Center,
        modifier = Modifier.fillMaxWidth())
}

@Composable
private fun SegmentProgress(segments: Int, replay: Boolean) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(25.dp), color = Paper) {
        Column(Modifier.padding(18.dp)) {
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Text(if (replay) "Replay transfer progress" else "Observation progress", style = MaterialTheme.typography.titleMedium, color = Brown, modifier = Modifier.weight(1f))
                Text("$segments / 20", style = MaterialTheme.typography.titleLarge, color = OliveDeep)
            }
            Spacer(Modifier.height(13.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                repeat(20) { index ->
                    Box(Modifier.weight(1f).height(12.dp).clip(RoundedCornerShape(50)).background(if (index < segments) Olive else Line))
                }
            }
            Spacer(Modifier.height(10.dp))
            Text(if (replay) "The Pi verifies all received data before running the forecast." else "Only complete, valid segments count toward the result.", style = MaterialTheme.typography.bodyMedium, color = BrownMuted)
        }
    }
}

@Composable
private fun CompletedTestScreen(monitor: MonitorState, onDone: () -> Unit, onAgain: () -> Unit) {
    val context = LocalContext.current
    val result = monitor.forecast
    val needsSupport = result?.isElevated() == true
    val preferences = remember { context.getSharedPreferences("emergency_contacts", 0) }
    var showEmergencyContacts by remember { mutableStateOf(false) }
    val emergencyMessage = result?.let {
        if (monitor.mode == SourceMode.LIVE)
            "AFib Forecast recorded an elevated model score of ${String.format(Locale.US, "%.3f", it.score)} from my ECG after a 10-minute test. This research result is not a diagnosis. Please contact me."
        else
            "AFib Forecast replayed a prerecorded ECG (${monitor.recordingId ?: "unknown"}) with an elevated model score of ${String.format(Locale.US, "%.3f", it.score)}. This was a simulation, not my ECG or a diagnosis."
    }.orEmpty()
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp).padding(top = 18.dp, bottom = 28.dp)) {
        Box(Modifier.size(64.dp).clip(CircleShape).background(if (needsSupport) RiskPale else Olive)
            .clickable(role = Role.Button, onClick = onDone)
            .semantics { contentDescription = "Return to home" }, contentAlignment = Alignment.Center) {
            AppGlyph(Glyph.CHECK, if (needsSupport) ErrorRed else Brown, Modifier.size(30.dp))
        }
        Spacer(Modifier.height(18.dp))
        Text(if (needsSupport) "Elevated result." else "Test complete.",
            style = MaterialTheme.typography.displayMedium, color = if (needsSupport) ErrorRed else Brown)
        Spacer(Modifier.height(6.dp))
        Text("Your result has been saved with this session.", style = MaterialTheme.typography.bodyLarge, color = BrownMuted)
        Spacer(Modifier.height(22.dp))
        ForecastCard(monitor)
        if (needsSupport) {
            Spacer(Modifier.height(14.dp))
            EmergencySupportCard(
                replay = monitor.mode != SourceMode.LIVE,
                onNotify = {
                    val family = preferences.getString("family", "").orEmpty()
                    val clinic = preferences.getString("clinic", "").orEmpty()
                    val recipients = listOf(family, clinic).filter { it.isNotBlank() }.joinToString(",")
                    if (recipients.isBlank()) showEmergencyContacts = true
                    else launchEmergencySms(context, recipients, emergencyMessage)
                },
                onEdit = { showEmergencyContacts = true },
                onCall = { context.startActivity(Intent(Intent.ACTION_DIAL, Uri.parse("tel:112"))) },
            )
        }
        Spacer(Modifier.height(16.dp))
        Button(
            onClick = onDone, modifier = Modifier.fillMaxWidth().heightIn(min = 64.dp),
            shape = RoundedCornerShape(22.dp), colors = ButtonDefaults.buttonColors(containerColor = Brown, contentColor = Paper),
        ) { Text("Return to home", style = MaterialTheme.typography.titleMedium) }
        Spacer(Modifier.height(10.dp))
        OutlinedButton(
            onClick = onAgain, modifier = Modifier.fillMaxWidth().heightIn(min = 58.dp),
            shape = RoundedCornerShape(20.dp), border = BorderStroke(1.5.dp, Brown.copy(alpha = .3f)),
        ) { Text("Start another test", style = MaterialTheme.typography.labelLarge, color = Brown) }
    }
    if (showEmergencyContacts && result != null) {
        EmergencyContactsDialog(
            initialFamily = preferences.getString("family", "").orEmpty(),
            initialClinic = preferences.getString("clinic", "").orEmpty(),
            onDismiss = { showEmergencyContacts = false },
            onPrepareMessage = { family, clinic ->
                preferences.edit().putString("family", family).putString("clinic", clinic).apply()
                val recipients = listOf(family, clinic).filter { it.isNotBlank() }.joinToString(",")
                if (recipients.isNotBlank()) launchEmergencySms(context, recipients, emergencyMessage)
                showEmergencyContacts = false
            },
        )
    }
}

@Composable
private fun EmergencySupportCard(replay: Boolean, onNotify: () -> Unit, onEdit: () -> Unit, onCall: () -> Unit) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(26.dp), color = RiskPale) {
        Column(Modifier.padding(20.dp)) {
            Text(if (replay) "Pre-recorded ECG result" else "Result needs review",
                style = MaterialTheme.typography.headlineMedium, color = ErrorRed)
            Spacer(Modifier.height(5.dp))
            Text(if (replay) "This elevated score came from a replay file, not your own ECG. It is not a personal health alert."
                else "This score crossed the configured research threshold. It is not a diagnosis or proof of an emergency.",
                style = MaterialTheme.typography.bodyLarge, color = BrownMuted)
            Spacer(Modifier.height(14.dp))
            Button(
                onClick = onNotify, modifier = Modifier.fillMaxWidth().heightIn(min = 58.dp),
                shape = RoundedCornerShape(20.dp), colors = ButtonDefaults.buttonColors(containerColor = ErrorRed, contentColor = Paper),
            ) { Text("Emergency contact", style = MaterialTheme.typography.titleMedium) }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                TextButton(onClick = onEdit) { Text("Edit contacts", color = Brown) }
                TextButton(onClick = onCall) { Text("Call emergency services", color = ErrorRed) }
            }
        }
    }
}

private fun launchEmergencySms(context: Context, recipients: String, message: String) {
    val normalized = recipients.split(',', ';').map { it.trim() }.filter { it.isNotBlank() }.joinToString(";")
    val intent = Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:$normalized")).putExtra("sms_body", message)
    runCatching { context.startActivity(intent) }
}

@Composable
private fun EmergencyContactsDialog(
    initialFamily: String,
    initialClinic: String,
    onDismiss: () -> Unit,
    onPrepareMessage: (String, String) -> Unit,
) {
    var family by remember { mutableStateOf(initialFamily) }
    var clinic by remember { mutableStateOf(initialClinic) }
    Dialog(onDismissRequest = onDismiss) {
        Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(28.dp), color = Paper) {
            Column(Modifier.verticalScroll(rememberScrollState()).padding(22.dp)) {
                Text("Emergency contacts", style = MaterialTheme.typography.headlineMedium, color = Brown)
                Spacer(Modifier.height(6.dp))
                Text("Add trusted family and a clinic or hospital number. Separate multiple numbers with commas.", style = MaterialTheme.typography.bodyMedium, color = BrownMuted)
                Spacer(Modifier.height(16.dp))
                OutlinedTextField(family, { family = it }, Modifier.fillMaxWidth(), label = { Text("Family or caregiver") }, singleLine = true)
                Spacer(Modifier.height(10.dp))
                OutlinedTextField(clinic, { clinic = it }, Modifier.fillMaxWidth(), label = { Text("Clinic or hospital") }, singleLine = true)
                Spacer(Modifier.height(16.dp))
                Button(
                    onClick = { onPrepareMessage(family, clinic) },
                    enabled = family.isNotBlank() || clinic.isNotBlank(),
                    modifier = Modifier.fillMaxWidth().heightIn(min = 58.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = ActionBrown, contentColor = Paper),
                    shape = RoundedCornerShape(20.dp),
                ) { Text("Save and prepare message", style = MaterialTheme.typography.titleMedium) }
                TextButton(onClick = onDismiss, modifier = Modifier.align(Alignment.CenterHorizontally)) { Text("Cancel", color = BrownMuted) }
                Text("Your SMS app opens with the message prepared. Review it and tap Send.", style = MaterialTheme.typography.labelMedium, color = BrownMuted, textAlign = TextAlign.Center)
            }
        }
    }
}

@Composable
private fun EmergencyContactSettings(onDismiss: () -> Unit) {
    val context = LocalContext.current
    val preferences = remember { context.getSharedPreferences("emergency_contacts", 0) }
    var family by remember { mutableStateOf(preferences.getString("family", "").orEmpty()) }
    var clinic by remember { mutableStateOf(preferences.getString("clinic", "").orEmpty()) }
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(Modifier.fillMaxWidth().padding(20.dp).heightIn(max = 700.dp),
            shape = RoundedCornerShape(28.dp), color = Paper) {
            Column(Modifier.verticalScroll(rememberScrollState()).padding(22.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Emergency contacts", style = MaterialTheme.typography.headlineMedium, color = Brown,
                        modifier = Modifier.weight(1f))
                    TextButton(onClick = onDismiss) { Text("Close", color = Brown) }
                }
                Spacer(Modifier.height(18.dp))
                Text("The app opens your SMS app for review; it does not send a message automatically.",
                    style = MaterialTheme.typography.bodyMedium, color = BrownMuted)
                Spacer(Modifier.height(10.dp))
                OutlinedTextField(family, { family = it }, Modifier.fillMaxWidth(),
                    label = { Text("Family or caregiver number") }, singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Phone))
                Spacer(Modifier.height(10.dp))
                OutlinedTextField(clinic, { clinic = it }, Modifier.fillMaxWidth(),
                    label = { Text("Clinic number (optional)") }, singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Phone))
                Spacer(Modifier.height(14.dp))
                Button(onClick = {
                    preferences.edit().putString("family", family.trim()).putString("clinic", clinic.trim()).apply()
                    onDismiss()
                }, modifier = Modifier.fillMaxWidth().heightIn(min = 54.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = ActionBrown, contentColor = Paper),
                    shape = RoundedCornerShape(18.dp)) {
                    Text("Save contacts", style = MaterialTheme.typography.titleMedium)
                }
            }
        }
    }
}

@Composable
private fun DetailLine(label: String, value: String) {
    Row(Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Text(label, style = MaterialTheme.typography.bodyMedium, color = BrownMuted, modifier = Modifier.weight(1f))
        Text(value, style = MaterialTheme.typography.labelLarge, color = Brown, textAlign = TextAlign.End)
    }
}

private fun formatTime(seconds: Int) = "%d:%02d".format(seconds / 60, seconds % 60)

@Composable
private fun HistoryScreen(history: List<SessionSummary>) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp).padding(bottom = 30.dp)) {
        Text("Sessions", style = MaterialTheme.typography.displayMedium, color = Brown)
        Spacer(Modifier.height(8.dp))
        Text("Previous completed tests and interruptions.", style = MaterialTheme.typography.bodyLarge, color = BrownMuted)
        Spacer(Modifier.height(14.dp))
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            SessionLegendItem("Live testing", Olive)
            Spacer(Modifier.width(18.dp))
            SessionLegendItem("AF simulation", Orange)
        }
        Spacer(Modifier.height(18.dp))
        history.forEach { session ->
            SessionCard(session)
            Spacer(Modifier.height(12.dp))
        }
    }
}

@Composable
private fun SessionLegendItem(label: String, color: Color) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(11.dp).clip(CircleShape).background(color))
        Spacer(Modifier.width(7.dp))
        Text(label, style = MaterialTheme.typography.labelMedium, color = BrownMuted)
    }
}

@Composable
private fun SessionCard(session: SessionSummary) {
    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(22.dp), color = Paper) {
        Row(Modifier.padding(horizontal = 18.dp, vertical = 16.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier.size(12.dp).clip(CircleShape).background(
                    when {
                        session.interrupted -> ErrorRed
                        session.mode == SourceMode.LIVE -> Olive
                        session.mode == SourceMode.REPLAY -> Orange
                        else -> Lavender
                    }
                )
            )
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(session.label, style = MaterialTheme.typography.titleMedium, color = Brown)
                Text(session.date, style = MaterialTheme.typography.bodyMedium, color = BrownMuted)
            }
            Column(horizontalAlignment = Alignment.End) {
                if (session.score != null) Text(String.format(Locale.US, "%.3f", session.score), style = MaterialTheme.typography.titleLarge, color = Brown)
                Text(session.duration, style = MaterialTheme.typography.labelMedium, color = BrownMuted)
                Text(session.outcome, style = MaterialTheme.typography.labelMedium, color = if (session.interrupted) ErrorRed else OliveDeep)
            }
        }
    }
}

@Composable
private fun AboutScreen() {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())
        .padding(horizontal = 20.dp).padding(bottom = 30.dp)) {
        Text("About AFib Forecast", style = MaterialTheme.typography.displayMedium, color = Brown)
        Spacer(Modifier.height(8.dp))
        Text("Single-lead ECG forecasting on a Raspberry Pi.",
            style = MaterialTheme.typography.bodyLarge, color = BrownMuted)
        Spacer(Modifier.height(22.dp))

        Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(24.dp), color = Paper) {
            Column(Modifier.padding(20.dp)) {
                Text("How it works", style = MaterialTheme.typography.titleLarge, color = Brown)
                Spacer(Modifier.height(8.dp))
                Text("ECG device → Raspberry Pi → this app. The Pi processes the ECG and sends the result here.",
                    style = MaterialTheme.typography.bodyLarge, color = BrownMuted)
                HorizontalDivider(Modifier.padding(vertical = 18.dp), color = Line)
                Text("Live testing", style = MaterialTheme.typography.titleMedium, color = Brown)
                Text("Collects 10 minutes of ECG before forecasting.",
                    style = MaterialTheme.typography.bodyMedium, color = BrownMuted)
                Spacer(Modifier.height(14.dp))
                Text("Prerecorded replay", style = MaterialTheme.typography.titleMedium, color = Brown)
                Text("Analyzes a saved 10-minute ECG recording, transferred faster than real time.",
                    style = MaterialTheme.typography.bodyMedium, color = BrownMuted)
            }
        }
        Spacer(Modifier.height(14.dp))
        Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(24.dp), color = SagePale) {
            Column(Modifier.padding(20.dp)) {
                Text("About the forecast", style = MaterialTheme.typography.titleMedium, color = Brown)
                Spacer(Modifier.height(5.dp))
                Text("The score concerns AF onset within the following 20 minutes. It does not detect current AF.",
                    style = MaterialTheme.typography.bodyMedium, color = Brown)
            }
        }
        Spacer(Modifier.height(20.dp))
        Text("Research prototype only. Not a medical device or a basis for treatment decisions.",
            style = MaterialTheme.typography.bodyMedium, color = BrownMuted)
    }
}

@Composable
private fun HelpDialog(onDismiss: () -> Unit) {
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Box(Modifier.fillMaxSize().padding(horizontal = 24.dp, vertical = 36.dp), contentAlignment = Alignment.Center) {
            Surface(
                Modifier.fillMaxWidth().widthIn(max = 560.dp).heightIn(max = 720.dp),
                shape = RoundedCornerShape(28.dp), color = Paper,
            ) {
                Column(Modifier.verticalScroll(rememberScrollState()).padding(24.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        BrandMark(Modifier.size(48.dp))
                        Spacer(Modifier.width(12.dp))
                        Text("How to use it", style = MaterialTheme.typography.headlineMedium, color = Brown, modifier = Modifier.weight(1f))
                        TextButton(onClick = onDismiss) { Text("Close", color = Brown) }
                    }
                    Spacer(Modifier.height(18.dp))
                    HelpStep("1", "Prepare safely", "Use an isolated, battery-powered ECG device. Do not charge or attach USB cables while electrodes are on the body.")
                    HelpStep("2", "Choose a mode", "Select Live testing or AF simulation. The app displays the mode reported by the Pi.")
                    HelpStep("3", "Connect to the Pi", "Put the phone and Pi on the same Wi-Fi, then enter the Pi address in Connection.")
                    HelpStep("4", "Replay a recording", "Choose POS102 or NEG024. The ESP32 sends the saved ECG to the Pi for verification and scoring.")
                    HelpStep("5", "Live observation", "A live forecast requires a complete 10-minute ECG window before a result can appear.")
                    Spacer(Modifier.height(8.dp))
                    Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(20.dp), color = SagePale) {
                        Text("This research app is not a medical device. For severe symptoms or an immediate emergency, contact local emergency services.", Modifier.padding(16.dp), style = MaterialTheme.typography.bodyMedium, color = Brown)
                    }
                }
            }
        }
    }
}

@Composable
private fun HelpStep(number: String, title: String, detail: String) {
    Row(Modifier.fillMaxWidth().padding(bottom = 16.dp), verticalAlignment = Alignment.Top) {
        Box(Modifier.size(38.dp).clip(CircleShape).background(ActionBrown), contentAlignment = Alignment.Center) {
            Text(number, style = MaterialTheme.typography.labelLarge, color = Paper)
        }
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.titleMedium, color = Brown)
            Text(detail, style = MaterialTheme.typography.bodyMedium, color = BrownMuted)
        }
    }
}

@Composable
private fun ConnectionDialog(
    settings: ConnectionSettings,
    isConnecting: Boolean,
    onDismiss: () -> Unit,
    onUrlChange: (String) -> Unit,
    onFixtureChange: (Boolean) -> Unit,
) {
    Dialog(onDismissRequest = onDismiss, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Box(
            Modifier.fillMaxSize().padding(horizontal = 24.dp, vertical = 36.dp),
            contentAlignment = Alignment.Center,
        ) {
        Surface(
            Modifier.fillMaxWidth().widthIn(max = 560.dp).heightIn(max = 720.dp),
            shape = RoundedCornerShape(28.dp),
            color = Paper,
        ) {
            Column(Modifier.verticalScroll(rememberScrollState()).padding(24.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Connection", style = MaterialTheme.typography.headlineMedium, color = Brown, modifier = Modifier.weight(1f))
                    Surface(Modifier.size(45.dp).clickable(role=Role.Button,onClick=onDismiss), shape=CircleShape, color=Cream) {
                        Box(contentAlignment=Alignment.Center) { Text("×", fontSize=30.sp, color=Brown, textAlign=TextAlign.Center) }
                    }
                }
                Spacer(Modifier.height(8.dp))
                Text("The phone connects to the Raspberry Pi over the same trusted local Wi-Fi or hotspot.", style = MaterialTheme.typography.bodyMedium, color = BrownMuted)
                Spacer(Modifier.height(18.dp))
                Text("Pi address", style = MaterialTheme.typography.labelLarge, color = Brown)
                Spacer(Modifier.height(6.dp))
                OutlinedTextField(
                    value = settings.baseUrl,
                    onValueChange = onUrlChange,
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    textStyle = MaterialTheme.typography.bodyLarge,
                    shape = RoundedCornerShape(18.dp),
                    colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = OliveDeep, unfocusedBorderColor = Line),
                    supportingText = { Text("Example: http://192.168.4.1:8080") },
                )
                Spacer(Modifier.height(12.dp))
                Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(20.dp)).background(Cream).padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("Presentation simulation", style = MaterialTheme.typography.titleMedium, color = Brown)
                        Text("Uses a labeled prerecorded fixture when hardware is unavailable.", style = MaterialTheme.typography.bodyMedium, color = BrownMuted)
                    }
                    Switch(
                        checked = settings.fixtureMode,
                        onCheckedChange = onFixtureChange,
                        colors = SwitchDefaults.colors(checkedThumbColor = Paper, checkedTrackColor = OliveDeep),
                    )
                }
                Spacer(Modifier.height(18.dp))
                Button(
                    onClick = { onFixtureChange(false) },
                    enabled = !isConnecting,
                    modifier = Modifier.fillMaxWidth().heightIn(min = 58.dp),
                    shape = RoundedCornerShape(20.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Brown, contentColor = Paper),
                ) {
                    if (isConnecting) CircularProgressIndicator(Modifier.size(22.dp), color = Paper, strokeWidth = 3.dp)
                    else AppGlyph(Glyph.CONNECTION, Paper, Modifier.size(23.dp))
                    Spacer(Modifier.width(10.dp))
                    Text(if (isConnecting) "Connecting…" else "Connect equipment", style = MaterialTheme.typography.labelLarge)
                }
                Spacer(Modifier.height(10.dp))
                Text("The ECG module communicates through the Raspberry Pi.", style = MaterialTheme.typography.labelMedium, color = BrownMuted, textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth())
            }
        }
        }
    }
}
