package com.afibforecast.app.data

import com.afibforecast.app.model.SessionPhase
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FixtureEngineTest {
    @Test fun waveformIsBoundedAndPopulated() {
        val values = FixtureEngine.waveform(12)
        assertEquals(260, values.size)
        assertTrue(values.max() < 1.2f)
        assertTrue(values.min() > -0.6f)
    }

    @Test fun simulationNeverExceedsTwentySegments() {
        val state = FixtureEngine.simulation(10_000)
        assertTrue(state.acceptedSegments in 0..20)
        if (state.phase == SessionPhase.RESULT) assertEquals(20, state.acceptedSegments)
    }
}
