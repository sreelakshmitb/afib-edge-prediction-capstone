package com.afibforecast.app.data

import com.afibforecast.app.model.MonitorState
import org.json.JSONObject
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

class PiClient(
    private val onEvent: (MonitorState) -> Unit,
    private val onConnectionChanged: (Boolean, String?) -> Unit,
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(4, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .pingInterval(15, TimeUnit.SECONDS)
        .build()
    private var socket: WebSocket? = null
    private var current = MonitorState()
    @Volatile private var generation = 0

    fun connect(baseUrl: String, previous: MonitorState) {
        disconnect()
        // Keep the last accepted sequence and progress across a temporary Wi-Fi drop.
        // A genuinely new Pi session is reset by PiEventParser when its session_id changes.
        current = previous
        val attempt = generation
        val httpBase = baseUrl.trimEnd('/')
        val wsBase = httpBase.replaceFirst("http://", "ws://").replaceFirst("https://", "wss://")
        onConnectionChanged(false, null)
        val healthRequest = Request.Builder().url("$httpBase/api/v1/status").get().build()
        client.newCall(healthRequest).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                if (attempt == generation) onConnectionChanged(false, e.message ?: "Pi health check failed")
            }
            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (attempt != generation) return
                    if (!it.isSuccessful) {
                        onConnectionChanged(false, "Pi returned HTTP ${it.code}")
                        return
                    }
                    openSocket(wsBase, attempt)
                }
            }
        })
    }

    private fun openSocket(wsBase: String, attempt: Int) {
        val request = Request.Builder().url("$wsBase/api/v1/events").build()
        socket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                if (attempt == generation) onConnectionChanged(true, null)
            }
            override fun onMessage(webSocket: WebSocket, text: String) {
                if (attempt != generation) return
                runCatching { PiEventParser.apply(text, current) }
                    .onSuccess { current = it; onEvent(it) }
            }
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                if (attempt == generation) onConnectionChanged(false, t.message ?: "Pi connection failed")
            }
            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                if (attempt == generation) onConnectionChanged(false, reason.ifBlank { "Pi stream closed" })
            }
        })
    }

    fun startReplay(baseUrl: String, recordingId: String, onError: (String) -> Unit) {
        val body = JSONObject().put("recording_id", recordingId).toString()
        val request = Request.Builder().url("${baseUrl.trimEnd('/')}/api/v1/replay/start")
            .post(body.toRequestBody("application/json".toMediaType())).build()
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) = onError(e.message ?: "Cannot start replay")
            override fun onResponse(call: Call, response: Response) {
                response.use { if (!it.isSuccessful) onError("Pi rejected replay: HTTP ${it.code}") }
            }
        })
    }

    fun disconnect() {
        generation++
        socket?.close(1000, "App disconnected")
        socket = null
    }
}
