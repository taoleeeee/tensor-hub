package com.taoleeeee.tensorhub.model

import kotlinx.serialization.Serializable

/**
 * Model metadata and configuration.
 */
@Serializable
data class ModelConfig(
    val id: String,
    val name: String,
    val type: ModelType,
    val url: String,
    val filename: String,
    val sizeBytes: Long = 0,
    val quantized: Boolean = true,
    val description: String = ""
)

@Serializable
enum class ModelType {
    WHISPER,
    EMBEDDING,
    CLASSIFICATION
}

/**
 * Built-in model registry. Add new models here.
 */
object ModelRegistry {

    val models = listOf(
        ModelConfig(
            id = "whisper-base",
            name = "Whisper Base",
            type = ModelType.WHISPER,
            url = "https://huggingface.co/litert-community/whisper-base/resolve/main/whisper_base_30s_f32.tflite",
            filename = "whisper-base.tflite",
            sizeBytes = 277_000_000,
            description = "Balanced speed and accuracy for transcription"
        ),
        ModelConfig(
            id = "bge-small-en-v1.5-q8",
            name = "BGE Small EN v1.5",
            type = ModelType.EMBEDDING,
            url = "https://huggingface.co/Bombek1/bge-small-en-v1.5-litert/resolve/main/BAAI_bge-small-en-v1.5.tflite",
            filename = "bge-small-en-v1.5.tflite",
            sizeBytes = 127_000_000,
            description = "384-dim embeddings, optimized for on-device inference"
        )
    )

    fun getById(id: String): ModelConfig? = models.find { it.id == id }
}
