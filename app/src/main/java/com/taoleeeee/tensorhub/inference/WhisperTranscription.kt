package com.taoleeeee.tensorhub.inference

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Complete Whisper transcription pipeline.
 *
 * Audio → MelSpectrogram → Encoder → Autoregreneous Decoder → Text
 *
 * Uses TFLite signature API for models with named "encode" and "decode" sub-graphs.
 * All tensor I/O uses direct ByteBuffers for zero-copy NNAPI delegation.
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

        // Causal attention mask values
        private const val MASK_ON = 0.0f
        private const val MASK_OFF = -1e9f
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

            // Dump tensor shapes for decode signature
            try {
                for (name in decIn) {
                    try {
                        val idx = interpreter.getInputIndex(name)
                        val tensor = interpreter.getInputTensor(idx)
                        Log.i(TAG, "Decode input '$name': idx=$idx, shape=${tensor.shape().contentToString()}, dataType=${tensor.dataType()}, numBytes=${tensor.numBytes()}")
                    } catch (e: Exception) {
                        Log.w(TAG, "Cannot introspect decode input '$name': ${e.message}")
                    }
                }
                for (name in decOut) {
                    try {
                        val idx = interpreter.getOutputIndex(name)
                        val tensor = interpreter.getOutputTensor(idx)
                        Log.i(TAG, "Decode output '$name': idx=$idx, shape=${tensor.shape().contentToString()}, dataType=${tensor.dataType()}, numBytes=${tensor.numBytes()}")
                    } catch (e: Exception) {
                        Log.w(TAG, "Cannot introspect decode output '$name': ${e.message}")
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Tensor introspection failed: ${e.message}")
            }

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
     * Autoregressive decoder loop.
     * Runs the decoder up to DECODER_MAX_TOKENS times, generating one token per step.
     *
     * Decoder inputs:
     *   - encoder_hidden_states: float32[1, 1500, 512] (fixed)
     *   - decoder_input_ids: int32[1, 128] (grows each step)
     *   - cache: float32[1, 1, 128, 128] (causal mask / KV cache)
     *
     * Decoder output:
     *   - logits: float32[1, 128, 51865] (vocab scores per position)
     */
    private fun decode(encoderOutput: ByteBuffer, language: String): IntArray {
        val sotSequence = tokenizer.buildSotSequence(language)
        val generatedTokens = mutableListOf<Int>()

        // Causal attention mask: lower triangular [1, 1, 128, 128]
        val cacheBuffer = buildCausalMask()

        // Build decoder input IDs — start with SOT sequence, pad to 128
        val decoderInputIds = IntArray(DECODER_MAX_TOKENS) { 0 }
        for (i in sotSequence.indices) {
            decoderInputIds[i] = sotSequence[i]
        }

        var step = sotSequence.size  // position of next token to predict

        // Autoregressive loop
        for (iteration in 0 until DECODER_MAX_TOKENS) {
            // Allocate direct ByteBuffer for decoder_input_ids: 128 × 4 bytes
            val idsBuffer = ByteBuffer.allocateDirect(DECODER_MAX_TOKENS * 4).apply {
                order(ByteOrder.nativeOrder())
                for (i in 0 until DECODER_MAX_TOKENS) putInt(decoderInputIds[i])
                rewind()
            }

            // Allocate output: 1 × 128 × 51865 × 4 bytes (logits)
            // This is ~25MB — allocate once and reuse
            val logitsBuffer = ByteBuffer.allocateDirect(1 * DECODER_MAX_TOKENS * VOCAB_SIZE * 4).apply {
                order(ByteOrder.nativeOrder())
            }

            // Run decoder — use discovered tensor names
            val decInputNames = interpreter.getSignatureInputs("decode")
            val decOutputNames = interpreter.getSignatureOutputs("decode")
            val decInputs = HashMap<String, Any>()
            decInputs[decInputNames[0]] = encoderOutput
            decInputs[decInputNames[1]] = idsBuffer
            decInputs[decInputNames[2]] = cacheBuffer
            val decOutputs = HashMap<String, Any>()
            decOutputs[decOutputNames[0]] = logitsBuffer
            interpreter.runSignature(decInputs, decOutputs, "decode")

            // Diagnostic: log tensor info on first iteration
            if (iteration == 0) {
                val logitsArr = FloatArray(10)
                logitsBuffer.position(0)
                val diagBB = ByteBuffer.allocate(40).order(ByteOrder.nativeOrder())
                logitsBuffer.get(diagBB.array())
                diagBB.rewind()
                for (i in 0 until 10) logitsArr[i] = diagBB.float
                Log.i(TAG, "Decode step 0: logits first 10 = ${logitsArr.contentToString()}")
                Log.i(TAG, "Decode step 0: logitsBuffer capacity=${logitsBuffer.capacity()}, step=$step")
                // Also check what the SOT sequence is
                Log.i(TAG, "SOT sequence: ${sotSequence.contentToString()}, step=$step")
            }

            // Greedy: find argmax at the current step position
            // Logits layout: [batch=1, seq=128, vocab=51865]
            // Offset to position `step`: step * VOCAB_SIZE * 4
            val offset = step * VOCAB_SIZE * 4
            logitsBuffer.position(offset)
            val stepLogits = FloatBuffer.wrap(FloatArray(VOCAB_SIZE))
            // Read from ByteBuffer into FloatBuffer
            val tempArr = ByteArray(VOCAB_SIZE * 4)
            logitsBuffer.position(offset)
            logitsBuffer.get(tempArr)
            val tempBB = ByteBuffer.wrap(tempArr).order(ByteOrder.nativeOrder())
            for (v in 0 until VOCAB_SIZE) {
                stepLogits.put(tempBB.float)
            }
            stepLogits.rewind()

            // Argmax
            var bestToken = 0
            var bestScore = Float.NEGATIVE_INFINITY
            var secondBest = 0
            var secondBestScore = Float.NEGATIVE_INFINITY
            for (v in 0 until VOCAB_SIZE) {
                val score = stepLogits.get()
                if (score > bestScore) {
                    secondBest = bestToken
                    secondBestScore = bestScore
                    bestScore = score
                    bestToken = v
                } else if (score > secondBestScore) {
                    secondBest = v
                    secondBestScore = score
                }
            }
            Log.d(TAG, "Step $iteration: argmax=$bestToken (score=$bestScore), 2nd=$secondBest (score=$secondBestScore), offset=$offset")

            // Check for end of text
            if (tokenizer.isEndOfText(bestToken)) {
                Log.d(TAG, "EOT at step $iteration, token=$bestToken")
                break
            }

            // Skip special tokens in output but still feed them back
            if (!tokenizer.isSpecial(bestToken)) {
                generatedTokens.add(bestToken)
            }

            // Append token to decoder input for next step
            if (step < DECODER_MAX_TOKENS - 1) {
                decoderInputIds[step] = bestToken
                step++
            } else {
                Log.d(TAG, "Max decoder length reached")
                break
            }
        }

        Log.d(TAG, "Decoded ${generatedTokens.size} text tokens in ${step} steps")
        return generatedTokens.toIntArray()
    }

    /**
     * Build causal attention mask: lower triangular float32[1, 1, 128, 128].
     * 0.0 for positions that can attend, -1e9 for masked positions.
     */
    private fun buildCausalMask(): ByteBuffer {
        val size = 1 * 1 * DECODER_MAX_TOKENS * DECODER_MAX_TOKENS * 4
        val buffer = ByteBuffer.allocateDirect(size)
        buffer.order(ByteOrder.nativeOrder())
        for (i in 0 until DECODER_MAX_TOKENS) {
            for (j in 0 until DECODER_MAX_TOKENS) {
                buffer.putFloat(if (j <= i) MASK_ON else MASK_OFF)
            }
        }
        buffer.rewind()
        return buffer
    }
}
