package com.taoleeeee.tensorhub.server.routes

import com.taoleeeee.tensorhub.inference.InferenceEngine
import com.taoleeeee.tensorhub.inference.WhisperTranscription
import com.taoleeeee.tensorhub.inference.WhisperNative
import com.taoleeeee.tensorhub.inference.AudioDecoder
import com.taoleeeee.tensorhub.inference.WhisperTokenizer
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
    private val nativePipelines = mutableMapOf<String, WhisperNative>()

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
        val engine = session.parms["engine"] ?: "native"
        val useNnapi = session.parms["use_nnapi"]?.toBoolean() ?: true

        val audioFile = File(tempFilePath)
        val text: String
        val durationSec: Float
        val tokenCount: Int
        val elapsed: Long

        try {
            val startTime = System.currentTimeMillis()

            if (engine == "native") {
                val modelFile = modelManager.getModelFile(model)
                    ?: throw Exception("Model file not found for $model")
                if (!modelFile.exists()) {
                    throw Exception("Model file is not downloaded for $model")
                }
                val vocabFile = modelManager.getVocabFile(model)
                    ?: throw Exception("Vocab file not found for $model. Download the model first.")

                val cacheKey = "$model-$useNnapi"
                val nativePipeline = nativePipelines.getOrPut(cacheKey) {
                    WhisperNative(modelFile, useNnapi)
                }

                // Decode WAV samples to 16kHz mono floats
                val samples = AudioDecoder.decode(audioFile).getOrThrow()
                durationSec = samples.size.toFloat() / 16000

                // Run native C++ Mel Spectrogram + Autoregressive Decoder
                val tokenIds = nativePipeline.transcribe(samples, language)
                tokenCount = tokenIds.size

                val tokenizer = WhisperTokenizer(vocabFile)
                text = tokenizer.decode(tokenIds)
            } else {
                // Check model is loaded in InferenceEngine (for Kotlin CPU/NNAPI interpreter)
                if (!inferenceEngine.isLoaded(model)) {
                    throw Exception("Model '$model' is not loaded. POST /v1/models/load first.")
                }
                val pipeline = transcriptionPipelines.getOrPut(model) {
                    val interpreter = inferenceEngine.getInterpreter(model)
                        ?: throw Exception("Failed to get interpreter for $model")
                    val vocabFile = modelManager.getVocabFile(model)
                        ?: throw Exception("Vocab file not found for $model. Download the model first.")
                    WhisperTranscription(interpreter, vocabFile)
                }

                val result = runBlocking { pipeline.transcribe(audioFile, language) }.getOrThrow()
                text = result.text
                durationSec = result.durationSeconds
                tokenCount = result.tokenCount
            }

            elapsed = System.currentTimeMillis() - startTime
        } catch (e: Exception) {
            audioFile.delete()
            return errorResponse(Response.Status.INTERNAL_ERROR, "Transcription failed: ${e.message}")
        }

        // Clean up temp file
        audioFile.delete()

        val response = when (responseFormat) {
            "text" -> NanoHTTPD.newFixedLengthResponse(
                Response.Status.OK, "text/plain", text
            )
            "verbose_json" -> {
                val json = buildJsonObject {
                    put("task", "transcribe")
                    put("language", language)
                    put("duration", durationSec.toDouble())
                    put("text", text)
                    put("engine", engine)
                    put("use_nnapi", useNnapi)
                    put("inference_time_ms", elapsed)
                    put("token_count", tokenCount)
                    put("segments", buildJsonArray {
                        add(buildJsonObject {
                            put("id", 0)
                            put("start", 0.0)
                            put("end", durationSec.toDouble())
                            put("text", text)
                        })
                    })
                }
                NanoHTTPD.newFixedLengthResponse(
                    Response.Status.OK, "application/json", json.toString()
                )
            }
            else -> { // "json"
                val json = buildJsonObject {
                    put("text", text)
                }
                NanoHTTPD.newFixedLengthResponse(
                    Response.Status.OK, "application/json", json.toString()
                )
            }
        }
        return response
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
