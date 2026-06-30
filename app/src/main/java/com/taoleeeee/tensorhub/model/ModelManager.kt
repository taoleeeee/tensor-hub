package com.taoleeeee.tensorhub.model

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.net.URL
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Manages model download, caching, loading and lifecycle.
 * Also handles auxiliary files like vocab.json for tokenizer support.
 */
class ModelManager(private val context: Context) {

    companion object {
        private const val TAG = "ModelManager"
    }

    data class ModelState(
        val config: ModelConfig,
        val downloaded: Boolean = false,
        val loaded: Boolean = false,
        val loading: Boolean = false
    )

    private val modelsDir = File(context.filesDir, "models").also { it.mkdirs() }

    private val _modelStates = MutableStateFlow<Map<String, ModelState>>(emptyMap())
    val modelStates: StateFlow<Map<String, ModelState>> = _modelStates

    init {
        // Scan for already-downloaded models on startup
        val states = mutableMapOf<String, ModelState>()
        for (config in ModelRegistry.models) {
            val file = File(modelsDir, config.filename)
            states[config.id] = ModelState(
                config = config,
                downloaded = file.exists()
            )
        }
        _modelStates.value = states
    }

    /**
     * Get the local file for a model.
     */
    fun getModelFile(modelId: String): File? {
        val config = ModelRegistry.getById(modelId) ?: return null
        return File(modelsDir, config.filename)
    }

    /**
     * Get the vocab file for a model (e.g., vocab.json for Whisper).
     * Returns null if the model doesn't require a vocab file.
     */
    fun getVocabFile(modelId: String): File? {
        val config = ModelRegistry.getById(modelId) ?: return null
        val vocabFilename = config.vocabFilename ?: return null
        val file = File(modelsDir, vocabFilename)
        return if (file.exists()) file else null
    }

    /**
     * Load a TFLite model into a MappedByteBuffer.
     */
    fun loadModel(modelId: String): MappedByteBuffer? {
        val file = getModelFile(modelId) ?: return null
        if (!file.exists()) {
            Log.w(TAG, "Model file not found: ${file.absolutePath}")
            return null
        }
        Log.i(TAG, "Loading model: $modelId (${file.length()} bytes)")
        val fis = FileInputStream(file)
        val channel = fis.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
    }

    /**
     * Check if a model is downloaded.
     */
    fun isDownloaded(modelId: String): Boolean {
        return getModelFile(modelId)?.exists() == true
    }

    /**
     * Check if a model's vocab file is downloaded.
     */
    fun isVocabDownloaded(modelId: String): Boolean {
        return getVocabFile(modelId) != null
    }

    /**
     * Download a model's vocab file if configured and not already present.
     */
    suspend fun downloadVocabIfNeeded(modelId: String): Result<Unit> = withContext(Dispatchers.IO) {
        val config = ModelRegistry.getById(modelId)
            ?: return@withContext Result.failure(Exception("Unknown model: $modelId"))
        val vocabUrl = config.vocabUrl
            ?: return@withContext Result.success(Unit) // No vocab needed
        val vocabFilename = config.vocabFilename
            ?: return@withContext Result.success(Unit)
        val targetFile = File(modelsDir, vocabFilename)

        if (targetFile.exists()) {
            Log.i(TAG, "Vocab already downloaded: $vocabFilename")
            return@withContext Result.success(Unit)
        }

        try {
            Log.i(TAG, "Downloading vocab: $vocabUrl")
            URL(vocabUrl).openStream().use { input ->
                targetFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            Log.i(TAG, "Vocab downloaded: $vocabFilename (${targetFile.length()} bytes)")
            Result.success(Unit)
        } catch (e: Exception) {
            Log.e(TAG, "Vocab download failed: ${e.message}")
            Result.failure(e)
        }
    }

    /**
     * Get model info as JSON-compatible map.
     */
    fun getModelInfo(modelId: String): Map<String, Any>? {
        val config = ModelRegistry.getById(modelId) ?: return null
        val file = getModelFile(modelId)
        return mapOf(
            "id" to config.id,
            "name" to config.name,
            "type" to config.type.name.lowercase(),
            "downloaded" to (file?.exists() == true),
            "size_bytes" to (file?.length() ?: config.sizeBytes),
            "description" to config.description,
            "vocab_downloaded" to isVocabDownloaded(modelId)
        )
    }

    /**
     * List all available models.
     */
    fun listModels(): List<Map<String, Any>> {
        return ModelRegistry.models.map { config ->
            val file = getModelFile(config.id)
            mapOf(
                "id" to config.id,
                "name" to config.name,
                "type" to config.type.name.lowercase(),
                "downloaded" to (file?.exists() == true),
                "size_bytes" to (file?.length() ?: config.sizeBytes),
                "description" to config.description,
                "vocab_downloaded" to isVocabDownloaded(config.id)
            )
        }
    }
}
