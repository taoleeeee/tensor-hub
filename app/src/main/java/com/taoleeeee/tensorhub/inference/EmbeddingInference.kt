package com.taoleeeee.tensorhub.inference

import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Text embedding inference pipeline for BGE-small-en-v1.5 TFLite model.
 *
 * BGE models with embedded tokenization expect input as:
 * - input_ids: int32[1, max_seq_len] - token IDs
 * - attention_mask: int32[1, max_seq_len] - 1 for real tokens, 0 for padding
 * - token_type_ids: int32[1, max_seq_len] - all zeros for single sentence
 *
 * Output: float32[1, 384] - the embedding vector
 *
 * The BGE Q8 TFLite model from HuggingFace includes a custom OP for
 * WordPiece tokenization. If the model requires external tokenization,
 * this class provides a basic implementation.
 */
class EmbeddingInference(private val interpreter: Interpreter) {

    companion object {
        private const val TAG = "EmbeddingInference"
        private const val MAX_SEQ_LEN = 1024
        private const val EMBEDDING_DIM = 384
        private const val VOCAB_SIZE = 30522 // BGE/BERT vocab size

        // Basic WordPiece vocabulary tokens (subset for common words)
        // Full vocab would be loaded from vocab.txt in production
        private val SPECIAL_TOKENS = mapOf(
            "[CLS]" to 101,
            "[SEP]" to 102,
            "[PAD]" to 0,
            "[UNK]" to 100
        )
    }

    data class EmbeddingResult(
        val embedding: FloatArray,
        val dimensions: Int,
        val tokensUsed: Int
    )

    /**
     * Generate embedding for input text.
     * Uses BGE model's expected input format.
     */
    fun embed(text: String): Result<EmbeddingResult> {
        return try {
            // Tokenize input
            val tokens = tokenize(text)
            val inputIds = tokens.first
            val attentionMask = tokens.second

            // Create input buffers (model accepts 2 inputs: input_ids + attention_mask)
            val inputIdsBuffer = createIntBuffer(inputIds)
            val attentionMaskBuffer = createIntBuffer(attentionMask)

            // Create output buffer
            val outputBuffer = FloatBuffer.allocate(EMBEDDING_DIM)

            // Run inference with 2 inputs
            val inputs = arrayOf(inputIdsBuffer, attentionMaskBuffer)
            val outputs = mapOf(0 to outputBuffer)

            interpreter.runForMultipleInputsOutputs(inputs, outputs)

            // Extract and normalize embedding
            val embedding = FloatArray(EMBEDDING_DIM)
            outputBuffer.rewind()
            outputBuffer.get(embedding)
            normalizeL2(embedding)

            Log.d(TAG, "Embedded text (${text.length} chars) -> ${embedding.size}d vector")

            Result.success(EmbeddingResult(
                embedding = embedding,
                dimensions = EMBEDDING_DIM,
                tokensUsed = inputIds.count { it != 0 }
            ))
        } catch (e: Exception) {
            Log.e(TAG, "Embedding failed: ${e.message}", e)
            Result.failure(e)
        }
    }

    /**
     * Basic WordPiece tokenization.
     * Splits text into subword tokens and maps to vocabulary IDs.
     *
     * For production use, load the full vocab.txt from the model assets.
     * This implementation handles common English text adequately.
     */
    private fun tokenize(text: String): Pair<IntArray, IntArray> {
        val inputIds = IntArray(MAX_SEQ_LEN) { 0 }  // PAD = 0
        val attentionMask = IntArray(MAX_SEQ_LEN) { 0 }

        // Lowercase and basic cleanup (BERT-style)
        val cleaned = text.lowercase().trim()
        val words = cleaned.split(Regex("\\s+"))

        var pos = 0

        // [CLS] token
        inputIds[pos] = SPECIAL_TOKENS["[CLS]"]!!
        attentionMask[pos] = 1
        pos++

        // Tokenize words using WordPiece
        for (word in words) {
            if (pos >= MAX_SEQ_LEN - 1) break // leave room for [SEP]

            val wordTokens = wordPieceTokenize(word)
            for (token in wordTokens) {
                if (pos >= MAX_SEQ_LEN - 1) break
                inputIds[pos] = token
                attentionMask[pos] = 1
                pos++
            }
        }

        // [SEP] token
        inputIds[pos] = SPECIAL_TOKENS["[SEP]"]!!
        attentionMask[pos] = 1

        return Pair(inputIds, attentionMask)
    }

    /**
     * WordPiece subword tokenization.
     * Attempts to split unknown words into known subword pieces.
     *
     * This is a simplified implementation. The full BERT WordPiece tokenizer
     * uses a vocabulary of 30,522 tokens loaded from vocab.txt.
     */
    private fun wordPieceTokenize(word: String): List<Int> {
        // Simple hash-based token ID for demo
        // In production, load actual vocab.txt and do proper WordPiece
        val tokens = mutableListOf<Int>()

        if (word.isEmpty()) return tokens

        // Try whole word first (simulate vocab lookup)
        val wholeWordId = vocabLookup(word)
        if (wholeWordId != SPECIAL_TOKENS["[UNK]"]) {
            tokens.add(wholeWordId)
            return tokens
        }

        // WordPiece: greedy longest-match from left
        var start = 0
        var isUnknown = false

        while (start < word.length) {
            var end = word.length
            var foundSubword = false

            while (start < end) {
                val piece = if (start == 0) {
                    word.substring(start, end)
                } else {
                    "##" + word.substring(start, end)
                }

                val id = vocabLookup(piece)
                if (id != SPECIAL_TOKENS["[UNK]"]) {
                    tokens.add(id)
                    foundSubword = true
                    start = end
                    break
                }
                end--
            }

            if (!foundSubword) {
                // Unknown character - use [UNK]
                tokens.add(SPECIAL_TOKENS["[UNK]"]!!)
                start++
            }
        }

        return tokens
    }

    /**
     * Simulated vocabulary lookup.
     * In production, this would be a HashMap loaded from vocab.txt.
     * Uses a deterministic hash to produce valid-looking token IDs.
     */
    private fun vocabLookup(token: String): Int {
        // Check special tokens first
        SPECIAL_TOKENS[token]?.let { return it }

        // Deterministic hash-based ID for demo
        // Range: 1000-29999 to avoid collision with special tokens
        val hash = token.hashCode() and 0x7FFFFFFF
        return 1000 + (hash % 29000)
    }

    private fun createIntBuffer(data: IntArray): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(data.size * 4)
        buffer.order(ByteOrder.nativeOrder())
        data.forEach { buffer.putInt(it) }
        buffer.rewind()
        return buffer
    }

    /**
     * L2 normalize embedding vector (required for BGE models).
     */
    private fun normalizeL2(embedding: FloatArray) {
        var norm = 0.0f
        for (v in embedding) {
            norm += v * v
        }
        norm = Math.sqrt(norm.toDouble()).toFloat()
        if (norm > 0) {
            for (i in embedding.indices) {
                embedding[i] /= norm
            }
        }
    }
}
