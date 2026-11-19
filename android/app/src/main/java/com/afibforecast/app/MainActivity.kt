package com.afibforecast.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.afibforecast.app.ui.AFibForecastApp
import com.afibforecast.app.ui.theme.AFibForecastTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AFibForecastTheme {
                val model: MainViewModel = viewModel()
                val state = model.state.collectAsStateWithLifecycle().value
                AFibForecastApp(
                    state = state,
                    onExperienceChange = model::chooseExperience,
                    onBaseUrlChange = model::updateBaseUrl,
                    onFixtureChange = model::setFixtureMode,
                    onReplay = model::replayFromStart,
                    onReplayStart = model::startReplay,
                    onRefresh = model::refresh,
                )
            }
        }
    }
}
