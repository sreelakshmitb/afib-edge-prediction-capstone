package com.afibforecast.app.data

import android.content.SharedPreferences
import com.afibforecast.app.model.SessionSummary
import com.afibforecast.app.model.SourceMode
import org.json.JSONArray
import org.json.JSONObject

/** A small, local record of completed tests; it is independent of the Pi connection. */
class SessionHistoryStore(private val prefs: SharedPreferences) {
    fun load(): List<SessionSummary> = runCatching {
        val items = JSONArray(prefs.getString(KEY, "[]"))
        (0 until items.length()).mapNotNull { index ->
            runCatching {
                val item = items.getJSONObject(index)
                SessionSummary(
                    id = item.getString("id"),
                    mode = SourceMode.valueOf(item.getString("mode")),
                    label = item.getString("label"),
                    date = item.getString("date"),
                    duration = item.getString("duration"),
                    outcome = item.getString("outcome"),
                    score = if (item.isNull("score")) null else item.optDouble("score").takeIf(Double::isFinite),
                    interrupted = item.optBoolean("interrupted", false),
                )
            }.getOrNull()
        }.take(MAX_ITEMS)
    }.getOrDefault(emptyList())

    fun save(history: List<SessionSummary>) {
        val items = JSONArray()
        history.take(MAX_ITEMS).forEach { entry ->
            items.put(JSONObject().apply {
                put("id", entry.id)
                put("mode", entry.mode.name)
                put("label", entry.label)
                put("date", entry.date)
                put("duration", entry.duration)
                put("outcome", entry.outcome)
                put("score", entry.score?.takeIf(Double::isFinite) ?: JSONObject.NULL)
                put("interrupted", entry.interrupted)
            })
        }
        prefs.edit().putString(KEY, items.toString()).commit()
    }

    private companion object {
        const val KEY = "completed_session_history"
        const val MAX_ITEMS = 100
    }
}
