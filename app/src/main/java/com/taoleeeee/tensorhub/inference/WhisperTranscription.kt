package com.taoleeeee.tensorhub.inference

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Complete Whisper transcription pipeline.
 *
 * Audio → MelSpectrogram → Encoder → Autoregressive Decoder → Text
 *
 * Uses TFLite signature API for models with named "encode" and "decode" sub-graphs.
 * All tensor I/O uses direct ByteBuffers for zero-copy NNAPI delegation.
 *
 * Decode logic follows Google's official LiteRT ASR sample:
 * https://github.com/google-ai-edge/litert-samples/tree/main/compiled_model_api/speech_recognition
 *
 * Runs on Dispatchers.Default to avoid blocking the main thread.
 */
class WhisperTranscription(
    private val interpreter: Interpreter,
    private val vocabFile: File
) {

    companion object {
        private const val TAG = "WhisperTranscription"

        // Encoder: mel spectrogram → hidden states
        private const val MEL_BANDS = 80
        private const val MEL_FRAMES = 3000
        private const val ENCODER_DIM = 512
        private const val ENCODER_SEQ_LEN = 1500

        // Decoder: autoregressive token generation
        private const val DECODER_MAX_TOKENS = 128
        private const val VOCAB_SIZE = 51865

        // Whisper special tokens
        private const val START_OF_TRANSCRIPT_TOKEN = 50258
        private const val END_OF_TEXT_TOKEN = 50257

        // Causal mask values — must match Google's reference implementation
        private const val MASKED_IN = 0.0f
        private const val MASKED_OUT = -0.7f * Float.MAX_VALUE  // ~-2.4e38, NOT -1e9
    }

    data class TranscriptionResult(
        val text: String,
        val language: String,
        val durationSeconds: Float,
        val tokenCount: Int,
        val inferenceTimeMs: Long
    )

    private val tokenizer = WhisperTokenizer(vocabFile)
    private var encoderInitialized = false

    /**
     * Transcribe audio from a WAV file.
     * Runs on a background thread — safe to call from coroutine context.
     */
    suspend fun transcribe(
        audioFile: File,
        language: String = "en"
    ): Result<TranscriptionResult> = withContext(Dispatchers.Default) {
        try {
            val startTime = System.currentTimeMillis()

            // 1. Decode audio to 16kHz mono float samples
            val samples = AudioDecoder.decode(audioFile).getOrThrow()
            val durationSec = samples.size.toFloat() / 16000

            // 2. Compute mel spectrogram [1, 80, 3000]
            val melSpec = MelSpectrogram.compute(samples)

            // 3. Encode: mel → encoder hidden states
            val encoderOutput = encode(melSpec)

            // 4. Decode: autoregressive loop
            val tokenIds = decode(encoderOutput, language)

            // 5. Decode tokens to text
            val text = tokenizer.decode(tokenIds)
            val elapsed = System.currentTimeMillis() - startTime

            Log.i(TAG, "Transcribed ${durationSec}s audio in ${elapsed}ms → \"${text.take(80)}\"")

            Result.success(TranscriptionResult(
                text = text,
                language = language,
                durationSeconds = durationSec,
                tokenCount = tokenIds.size,
                inferenceTimeMs = elapsed
            ))
        } catch (e: Exception) {
            Log.e(TAG, "Transcription failed: ${e.message}", e)
            Result.failure(e)
        }
    }

    /**
     * Run the encoder: float32[1,80,3000] → float32[1,1500,512]
     * Uses direct ByteBuffer for zero-copy NNAPI delegation.
     */
    private fun encode(melSpec: Array<Array<FloatArray>>): ByteBuffer {
        // Allocate direct ByteBuffer for input: 1 × 80 × 3000 × 4 bytes
        val inputSize = 1 * MEL_BANDS * MEL_FRAMES * 4
        val inputBuffer = ByteBuffer.allocateDirect(inputSize).apply {
            order(ByteOrder.nativeOrder())
        }
        for (m in 0 until MEL_BANDS) {
            for (t in 0 until MEL_FRAMES) {
                inputBuffer.putFloat(melSpec[0][m][t])
            }
        }
        inputBuffer.rewind()

        // Allocate direct ByteBuffer for output: 1 × 1500 × 512 × 4 bytes
        val outputSize = 1 * ENCODER_SEQ_LEN * ENCODER_DIM * 4
        val outputBuffer = ByteBuffer.allocateDirect(outputSize).apply {
            order(ByteOrder.nativeOrder())
        }

        // Discover tensor names on first run
        if (!encoderInitialized) {
            val encIn = interpreter.getSignatureInputs("encode")
            val encOut = interpreter.getSignatureOutputs("encode")
            Log.i(TAG, "Encode inputs: ${encIn.contentToString()}, outputs: ${encOut.contentToString()}")
            val decIn = interpreter.getSignatureInputs("decode")
            val decOut = interpreter.getSignatureOutputs("decode")
            Log.i(TAG, "Decode inputs: ${decIn.contentToString()}, outputs: ${decOut.contentToString()}")
            encoderInitialized = true
        }

        // Run encoder via runSignature(inputs, outputs, signatureName)
        val encInputs = HashMap<String, Any>()
        encInputs[interpreter.getSignatureInputs("encode")[0]] = inputBuffer
        val encOutputs = HashMap<String, Any>()
        encOutputs[interpreter.getSignatureOutputs("encode")[0]] = outputBuffer
        interpreter.runSignature(encInputs, encOutputs, "encode")

        Log.d(TAG, "Encoder output ready (${outputSize} bytes)")
        return outputBuffer
    }

    /**
     * Autoregressive decoder loop — follows Google's LiteRT ASR sample + Whisper prompt format.
     *
     * Decode signature:
     *   inputs:  float32[1,1500,512] (encoder output), int32[1,128] (token ids), float32[1,1,128,128] (causal mask)
     *   outputs: float32[1,128,51865] (logits)
     *
     * Uses full Whisper SOT prompt: <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
     * Causal mask uses -0.7f * Float.MAX_VALUE per Google's reference.
     * Logits read as flat float array, argmax at step * VOCAB_SIZE.
     */
    private fun decode(encoderOutput: ByteBuffer, language: String): IntArray {
        val sotSequence = tokenizer.buildSotSequence(language)
        val generatedTokens = mutableListOf<Int>()
        val decInputNames = interpreter.getSignatureInputs("decode")
        val decOutputNames = interpreter.getSignatureOutputs("decode")

        // Build causal mask: lower triangular [1, 1, 128, 128]
        // Matches Google's reference: MASKED_IN=0.0, MASKED_OUT=-0.7f*MAX_VALUE
        val maskSize = DECODER_MAX_TOKENS * DECODER_MAX_TOKENS
        val causalMask = FloatArray(maskSize) { MASKED_OUT }
        for (r in 0 until DECODER_MAX_TOKENS) {
            for (c in 0..r) {
                causalMask[r * DECODER_MAX_TOKENS + c] = MASKED_IN
            }
        }
        val maskBuffer = ByteBuffer.allocateDirect(maskSize * 4).apply {
            order(ByteOrder.nativeOrder())
            for (v in causalMask) putFloat(v)
            rewind()
        }

        // Token IDs: full SOT sequence at start, rest zeros
        val tokenIds = IntArray(DECODER_MAX_TOKENS) { 0 }
        for (i in sotSequence.indices) {
            tokenIds[i] = sotSequence[i]
        }
        var step = sotSequence.size  // position of next token to predict

        Log.i(TAG, "SOT sequence: ${sotSequence.contentToString()}, starting at step=$step")

        // Output logits buffer: [1, 128, 51865] = 128 * 51865 floats
        val numLogits = DECODER_MAX_TOKENS * VOCAB_SIZE

        // Autoregressive loop
        for (iteration in 0 until DECODER_MAX_TOKENS) {
            // Write token IDs to direct ByteBuffer
            val idsBuffer = ByteBuffer.allocateDirect(DECODER_MAX_TOKENS * 4).apply {
                order(ByteOrder.nativeOrder())
                for (id in tokenIds) putInt(id)
                rewind()
            }

            // Allocate logits output buffer
            val logitsBuffer = ByteBuffer.allocateDirect(numLogits * 4).apply {
                order(ByteOrder.nativeOrder())
            }

            // Run decoder
            val decInputs = HashMap<String, Any>()
            decInputs[decInputNames[0]] = encoderOutput
            decInputs[decInputNames[1]] = idsBuffer
            decInputs[decInputNames[2]] = maskBuffer
            val decOutputs = HashMap<String, Any>()
            decOutputs[decOutputNames[0]] = logitsBuffer
            interpreter.runSignature(decInputs, decOutputs, "decode")

            // Read logits as flat float array
            val logits = FloatArray(numLogits)
            logitsBuffer.rewind()
            logitsBuffer.order(ByteOrder.nativeOrder())
            for (idx in 0 until numLogits) {
                logits[idx] = logitsBuffer.float
            }

            // Argmax at position `step` (the token we just predicted)
            val startIndex = step * VOCAB_SIZE
            val endIndex = (step + 1) * VOCAB_SIZE
            var bestToken = startIndex
            var bestScore = logits[startIndex]
            for (idx in startIndex + 1 until endIndex) {
                if (logits[idx] > bestScore) {
                    bestScore = logits[idx]
                    bestToken = idx
                }
            }
            val tokenId = bestToken - startIndex  // Convert flat index to token ID

            Log.d(TAG, "Step $iteration (pos=$step): token=$tokenId (score=${"%.2f".format(bestScore)})")

            // Check for end of text
            if (tokenId == END_OF_TEXT_TOKEN) {
                Log.d(TAG, "EOT at step $iteration")
                break
            }

            // Collect non-special tokens for output
            if (!tokenizer.isSpecial(tokenId)) {
                generatedTokens.add(tokenId)
            }

            // Feed predicted token back for next step
            if (step < DECODER_MAX_TOKENS - 1) {
                tokenIds[step] = tokenId
                step++
            } else {
                Log.d(TAG, "Max decoder length reached")
                break
            }
        }

        Log.d(TAG, "Decoded ${generatedTokens.size} text tokens in $step steps")
        return generatedTokens.toIntArray()
    }
}
