package com.taoleeeee.tensorhub.server.routes

import com.taoleeeee.tensorhub.inference.InferenceEngine
import com.taoleeeee.tensorhub.model.ModelManager
import fi.iki.elonen.NanoHTTPD
import fi.iki.elonen.NanoHTTPD.IHTTPSession
import fi.iki.elonen.NanoHTTPD.Response
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.*

/**
 * Handles /v1/models endpoints for model management.
 */
class ModelRoutes(
    private val inferenceEngine: InferenceEngine,
    private val modelManager: ModelManager
) {
    fun handleList(): Response {
        val models = buildJsonArray {
            modelManager.listModels().forEach { model ->
                add(buildJsonObject {
                    put("id", model["id"].toString())
                    put("object", "model")
                    put("owned_by", "tensor-hub")
                    put("loaded", inferenceEngine.isLoaded(model["id"].toString()))
                })
            }
        }

        val body = buildJsonObject {
            put("object", "list")
            put("data", models)
        }

        return NanoHTTPD.newFixedLengthResponse(Response.Status.OK, "application/json", body.toString())
    }

    fun handleLoad(session: IHTTPSession): Response {
        val bodyBytes = ByteArray(session.headers["content-length"]?.toIntOrNull() ?: 0)
        session.inputStream.read(bodyBytes)
        val bodyStr = String(bodyBytes)

        val body = try {
            json.parseToJsonElement(bodyStr) as? JsonObject
        } catch (e: Exception) {
            return errorResponse(Response.Status.BAD_REQUEST, "Invalid JSON body")
        }

        val modelId = body?.get("model")?.jsonPrimitive?.content
            ?: return errorResponse(Response.Status.BAD_REQUEST, "Missing 'model' field")

        // Run loading in a blocking way (server thread handles it)
        val result = runBlocking {
            // Download vocab file if needed (e.g., for Whisper tokenizer)
            modelManager.downloadVocabIfNeeded(modelId)
            inferenceEngine.loadModel(modelId)
        }

        return if (result.isSuccess) {
            val responseBody = buildJsonObject {
                put("status", "ok")
                put("model", modelId)
                put("delegate", inferenceEngine.getDelegateType(modelId)?.name ?: "unknown")
            }
            NanoHTTPD.newFixedLengthResponse(Response.Status.OK, "application/json", responseBody.toString())
        } else {
            errorResponse(Response.Status.INTERNAL_ERROR, result.exceptionOrNull()?.message ?: "Load failed")
        }
    }

    fun handleUnload(session: IHTTPSession): Response {
        val bodyBytes = ByteArray(session.headers["content-length"]?.toIntOrNull() ?: 0)
        session.inputStream.read(bodyBytes)
        val bodyStr = String(bodyBytes)

        val body = try {
            json.parseToJsonElement(bodyStr) as? JsonObject
        } catch (e: Exception) {
            return errorResponse(Response.Status.BAD_REQUEST, "Invalid JSON body")
        }

        val modelId = body?.get("model")?.jsonPrimitive?.content
            ?: return errorResponse(Response.Status.BAD_REQUEST, "Missing 'model' field")

        inferenceEngine.unloadModel(modelId)

        val responseBody = buildJsonObject {
            put("status", "ok")
            put("model", modelId)
        }
        return NanoHTTPD.newFixedLengthResponse(Response.Status.OK, "application/json", responseBody.toString())
    }

    companion object {
        private val json = Json { prettyPrint = true }
    }

    private fun errorResponse(status: Response.Status, message: String): Response {
        val errorObj = buildJsonObject {
            put("error", message)
        }
        return NanoHTTPD.newFixedLengthResponse(
            status,
            "application/json",
            errorObj.toString()
        )
    }
}
