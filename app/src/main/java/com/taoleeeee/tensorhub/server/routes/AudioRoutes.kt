package com.taoleeeee.tensorhub.server.routes

import com.taoleeeee.tensorhub.inference.InferenceEngine
import com.taoleeeee.tensorhub.inference.WhisperTranscription
import com.taoleeeee.tensorhub.model.ModelManager
import fi.iki.elonen.NanoHTTPD
import fi.iki.elonen.NanoHTTPD.IHTTPSession
import kotlinx.coroutines.runBlocking
import fi.iki.elonen.NanoHTTPD.Response
import kotlinx.serialization.json.*
import java.io.File

/**
 * Handles /v1/audio/transcriptions
 * OpenAI-compatible multipart form upload.
 *
 * Accepts WAV audio, runs through Whisper encoder-decoder pipeline,
 * returns transcription as JSON or plain text.
 */
class AudioRoutes(
    private val inferenceEngine: InferenceEngine,
    private val modelManager: ModelManager
) {

    companion object {
        private const val TAG = "AudioRoutes"
    }

    private val transcriptionPipelines = mutableMapOf<String, WhisperTranscription>()

    fun handleTranscription(session: IHTTPSession): Response {
        // Parse multipart body
        val files = mutableMapOf<String, String>()
        try {
            session.parseBody(files)
        } catch (e: Exception) {
            return errorResponse(Response.Status.BAD_REQUEST, "Failed to parse body: ${e.message}")
        }

        // Get uploaded file path (NanoHTTPD saves temp file)
        val tempFilePath = files["file"]
            ?: return errorResponse(Response.Status.BAD_REQUEST, "Missing 'file' field in multipart form")

        val model = session.parms["model"] ?: "whisper-base"
        val language = session.parms["language"] ?: "en"
        val responseFormat = session.parms["response_format"] ?: "json"

        // Check model is loaded
        if (!inferenceEngine.isLoaded(model)) {
            return errorResponse(
                Response.Status.BAD_REQUEST,
                "Model '$model' is not loaded. POST /v1/models/load first."
            )
        }

        // Get or create transcription pipeline
        val pipeline = transcriptionPipelines.getOrPut(model) {
            val interpreter = inferenceEngine.getInterpreter(model)
                ?: return errorResponse(Response.Status.INTERNAL_ERROR, "Failed to get interpreter for $model")
            val vocabFile = modelManager.getVocabFile(model)
                ?: return errorResponse(Response.Status.INTERNAL_ERROR, "Vocab file not found for $model. Download the model first.")
            WhisperTranscription(interpreter, vocabFile)
        }

        // Run transcription in blocking coroutine (NanoHTTPD runs on its own thread)
        val audioFile = File(tempFilePath)
        val result = runBlocking { pipeline.transcribe(audioFile, language) }

        // Clean up temp file
        audioFile.delete()

        return result.fold(
            onSuccess = { transcription ->
                val content = when (responseFormat) {
                    "text" -> NanoHTTPD.newFixedLengthResponse(
                        Response.Status.OK, "text/plain", transcription.text
                    )
                    "verbose_json" -> {
                        val json = buildJsonObject {
                            put("task", "transcribe")
                            put("language", transcription.language)
                            put("duration", transcription.durationSeconds.toDouble())
                            put("text", transcription.text)
                            put("segments", buildJsonArray {
                                add(buildJsonObject {
                                    put("id", 0)
                                    put("start", 0.0)
                                    put("end", transcription.durationSeconds.toDouble())
                                    put("text", transcription.text)
                                })
                            })
                        }
                        NanoHTTPD.newFixedLengthResponse(
                            Response.Status.OK, "application/json", json.toString()
                        )
                    }
                    else -> { // "json"
                        val json = buildJsonObject {
                            put("text", transcription.text)
                        }
                        NanoHTTPD.newFixedLengthResponse(
                            Response.Status.OK, "application/json", json.toString()
                        )
                    }
                }
                content
            },
            onFailure = { e ->
                errorResponse(Response.Status.INTERNAL_ERROR, "Transcription failed: ${e.message}")
            }
        )
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
