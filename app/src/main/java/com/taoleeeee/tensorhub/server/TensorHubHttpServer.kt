package com.taoleeeee.tensorhub.server

import android.util.Log
import com.taoleeeee.tensorhub.inference.InferenceEngine
import com.taoleeeee.tensorhub.model.ModelManager
import com.taoleeeee.tensorhub.server.routes.AudioRoutes
import com.taoleeeee.tensorhub.server.routes.EmbeddingRoutes
import com.taoleeeee.tensorhub.server.routes.HealthRoutes
import com.taoleeeee.tensorhub.server.routes.ModelRoutes
import fi.iki.elonen.NanoHTTPD

/**
 * HTTP server exposing OpenAI-compatible API endpoints.
 * Bound to 127.0.0.1 only - never exposed to external networks.
 */
class TensorHubHttpServer(
    host: String = "127.0.0.1",
    port: Int = 8190,
    private val inferenceEngine: InferenceEngine,
    private val modelManager: ModelManager
) : NanoHTTPD(host, port) {

    companion object {
        private const val TAG = "HttpServer"
    }

    private val healthRoutes = HealthRoutes(inferenceEngine, modelManager)
    private val audioRoutes = AudioRoutes(inferenceEngine, modelManager)
    private val embeddingRoutes = EmbeddingRoutes(inferenceEngine)
    private val modelRoutes = ModelRoutes(inferenceEngine, modelManager)

    override fun serve(session: IHTTPSession): Response {
        val uri = session.uri
        val method = session.method

        Log.d(TAG, "$method $uri")

        return try {
            when {
                // Health
                uri == "/health" && method == Method.GET -> healthRoutes.handle()

                // Audio transcription (OpenAI-compatible)
                uri == "/v1/audio/transcriptions" && method == Method.POST -> audioRoutes.handleTranscription(session)

                // Embeddings (OpenAI-compatible)
                uri == "/v1/embeddings" && method == Method.POST -> embeddingRoutes.handleEmbedding(session)

                // Model management
                uri == "/v1/models" && method == Method.GET -> modelRoutes.handleList()
                uri == "/v1/models/load" && method == Method.POST -> modelRoutes.handleLoad(session)
                uri == "/v1/models/unload" && method == Method.POST -> modelRoutes.handleUnload(session)

                // 404
                else -> newFixedLengthResponse(
                    Response.Status.NOT_FOUND,
                    "application/json",
                    """{"error": "Not found: $method $uri"}"""
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error handling $method $uri: ${e.message}", e)
            newFixedLengthResponse(
                Response.Status.INTERNAL_ERROR,
                "application/json",
                """{"error": "${e.message?.replace("\"", "'") ?: "Internal error"}"}"""
            )
        }
    }
}
