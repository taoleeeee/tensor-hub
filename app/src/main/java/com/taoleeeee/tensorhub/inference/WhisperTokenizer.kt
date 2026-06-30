package com.taoleeeee.tensorhub.inference

import android.util.Log
import kotlinx.serialization.json.*
import java.io.File

/**
 * Whisper BPE tokenizer — decodes token IDs to text.
 * Loads vocab.json at runtime from the model directory.
 * Uses GPT-2 byte-level encoding reversal.
 */
class WhisperTokenizer(private val vocabFile: File) {

    companion object {
        private const val TAG = "WhisperTokenizer"

        // Whisper-base special token IDs
        const val EOT = 50257
        const val SOT = 50258
        const val TRANSCRIBE = 50359
        const val NOTIMESTAMPS = 50363
        const val NOSPEECH = 50362

        // Range of special tokens (language tokens are 50259-50357)
        private val SPECIAL_TOKEN_RANGE = 50257..51864

        /**
         * Build GPT-2 byte decoder: maps a Unicode char back to the original byte.
         * GPT-2's byte_encoder maps bytes → printable Unicode chars for BPE operation.
         * We reverse that mapping here.
         */
        private fun buildByteDecoder(): Map<Char, Byte> {
            val decoder = mutableMapOf<Char, Byte>()
            // Printable ASCII (33-126) map to themselves
            for (b in 33..126) decoder[b.toChar()] = b.toByte()
            // Latin-1 supplement (161-172, 174-255) map to themselves
            for (b in 161..172) decoder[b.toChar()] = b.toByte()
            for (b in 174..255) decoder[b.toChar()] = b.toByte()
            // Remaining bytes (0-32, 127-160, 173) map to U+0100 onwards
            val remaining = (0..32).toList() + (127..160).toList() + listOf(173)
            for ((i, b) in remaining.withIndex()) {
                decoder[(0x0100 + i).toChar()] = b.toByte()
            }
            return decoder
        }
    }

    private val byteDecoder = buildByteDecoder()
    private var idToToken: Map<Int, String>? = null

    /** Load vocabulary lazily on first decode call. */
    private fun ensureLoaded(): Map<Int, String> {
        idToToken?.let { return it }
        Log.i(TAG, "Loading vocab from ${vocabFile.absolutePath} (${vocabFile.length()} bytes)")
        val jsonStr = vocabFile.readText()
        val jsonObj = Json.parseToJsonElement(jsonStr).jsonObject
        // vocab.json maps token-string → id; we reverse it
        val map = jsonObj.entries.associate { (token, id) -> id.jsonPrimitive.int to token }
        Log.i(TAG, "Loaded ${map.size} tokens")
        idToToken = map
        return map
    }

    /**
     * Decode an array of token IDs into a text string.
     * Filters out special tokens, applies GPT-2 byte decoding, returns UTF-8.
     */
    fun decode(tokenIds: IntArray): String {
        val vocab = ensureLoaded()
        val bytes = mutableListOf<Byte>()

        for (id in tokenIds) {
            // Skip special tokens
            if (id in SPECIAL_TOKEN_RANGE) continue

            val tokenStr = vocab[id] ?: continue
            for (ch in tokenStr) {
                byteDecoder[ch]?.let { bytes.add(it) }
            }
        }

        return String(bytes.toByteArray(), Charsets.UTF_8).trim()
    }

    /** Check if a token ID is the end-of-text token. */
    fun isEndOfText(tokenId: Int): Boolean = tokenId == EOT || tokenId == NOSPEECH

    /** Check if a token ID is special (not regular text). */
    fun isSpecial(tokenId: Int): Boolean = tokenId in SPECIAL_TOKEN_RANGE

    /** Build the initial decoder input sequence for English transcription. */
    fun buildSotSequence(language: String = "en"): IntArray {
        val langToken = 50259 + LANGUAGE_IDS.getOrDefault(language, 0)
        return intArrayOf(SOT, langToken, TRANSCRIBE, NOTIMESTAMPS)
    }

    /** Language code → offset from 50259. */
    private val LANGUAGE_IDS = mapOf(
        "en" to 0, "zh" to 1, "de" to 2, "es" to 3, "ru" to 4, "ko" to 5,
        "fr" to 6, "ja" to 7, "pt" to 8, "tr" to 9, "pl" to 10, "ca" to 11,
        "nl" to 12, "ar" to 13, "sv" to 14, "it" to 15, "id" to 16, "hi" to 17,
        "fi" to 18, "vi" to 19, "he" to 20, "uk" to 21, "el" to 22, "ms" to 23,
        "cs" to 24, "ro" to 25, "da" to 26, "hu" to 27, "ta" to 28, "no" to 29,
        "th" to 30, "ur" to 31, "hr" to 32, "bg" to 33, "lt" to 34, "la" to 35,
        "mi" to 36, "ml" to 37, "cy" to 38, "sk" to 39, "te" to 40, "fa" to 41,
        "lv" to 42, "bn" to 43, "sr" to 44, "az" to 45, "sl" to 46, "kn" to 47,
        "et" to 48, "mk" to 49, "br" to 50, "eu" to 51, "is" to 52, "hy" to 53,
        "ne" to 54, "mn" to 55, "bs" to 56, "kk" to 57, "sq" to 58, "sw" to 59,
        "gl" to 60, "mr" to 61, "pa" to 62, "si" to 63, "km" to 64, "sn" to 65,
        "yo" to 66, "so" to 67, "af" to 68, "oc" to 69, "ka" to 70, "be" to 71,
        "tg" to 72, "sd" to 73, "gu" to 74, "am" to 75, "yi" to 76, "lo" to 77,
        "uz" to 78, "fo" to 79, "ht" to 80, "ps" to 81, "tk" to 82, "nn" to 83,
        "mt" to 84, "sa" to 85, "lb" to 86, "my" to 87, "bo" to 88, "tl" to 89,
        "mg" to 90, "as" to 91, "tt" to 92, "haw" to 93, "ln" to 94, "ha" to 95,
        "ba" to 96, "jw" to 97, "su" to 98
    )
}
