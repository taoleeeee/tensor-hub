package com.taoleeeee.tensorhub.inference

import android.util.Log
import kotlin.math.*

/**
 * Computes Whisper-compatible mel spectrograms from raw audio samples.
 *
 * Whisper's preprocessing pipeline:
 * 1. STFT with n_fft=400, hop_length=160, Hann window
 * 2. Power spectrum (magnitude squared)
 * 3. 80-band mel filterbank (0-8000 Hz)
 * 4. Log10 → clamp to max-8.0 → normalize to [-1, 1]
 * 5. Pad/truncate to 3000 frames (30 seconds)
 *
 * Output shape: [1, 80, 3000] for the encoder.
 */
object MelSpectrogram {

    private const val TAG = "MelSpectrogram"
    private const val SAMPLE_RATE = 16000
    private const val N_FFT = 400
    private const val HOP_LENGTH = 160
    private const val N_MELS = 80
    private const val N_FRAMES = 3000  // 30 seconds at 16kHz with hop=160
    private const val F_MIN = 0.0f
    private const val F_MAX = 8000.0f

    // Precomputed: Hann window
    private val hannWindow = FloatArray(N_FFT) { i ->
        (0.5f * (1.0f - cos(2.0f * PI.toFloat() * i / N_FFT)))
    }

    // Precomputed: DFT twiddle factors for n_fft=400, first 201 bins
    // cos_table[k][n] = cos(2*pi*k*n/400), sin_table[k][n] = sin(2*pi*k*n/400)
    // Only compute positive frequencies: k = 0..200 (n_fft/2)
    private const val N_FREQ_BINS = N_FFT / 2 + 1  // 201
    private val cosTable: Array<FloatArray>
    private val sinTable: Array<FloatArray>

    // Precomputed: mel filterbank matrix [80][201]
    private val melFilterbank: Array<FloatArray>

    init {
        // Build DFT twiddle factor tables
        cosTable = Array(N_FREQ_BINS) { k ->
            FloatArray(N_FFT) { n ->
                cos(2.0f * PI.toFloat() * k * n / N_FFT)
            }
        }
        sinTable = Array(N_FREQ_BINS) { k ->
            FloatArray(N_FFT) { n ->
                -sin(2.0f * PI.toFloat() * k * n / N_FFT)
            }
        }

        // Build mel filterbank
        melFilterbank = buildMelFilterbank()
        Log.i(TAG, "Initialized: n_fft=$N_FFT, hop=$HOP_LENGTH, mels=$N_MELS, frames=$N_FRAMES")
    }

    /**
     * Compute mel spectrogram from raw 16kHz mono float samples.
     * Returns float array of shape [1, 80, 3000].
     */
    fun compute(samples: FloatArray): Array<Array<FloatArray>> {
        // 1. Frame the audio
        val nFrames = 1 + (samples.size - N_FFT) / HOP_LENGTH
        if (nFrames <= 0) {
            Log.w(TAG, "Audio too short (${samples.size} samples), padding to minimum")
            return compute(FloatArray(N_FFT) { if (it < samples.size) samples[it] else 0f })
        }

        // 2. Compute power spectrogram for each frame
        val spectrogram = Array(nFrames) { frameIdx ->
            val start = frameIdx * HOP_LENGTH
            val frame = FloatArray(N_FFT) { i ->
                val idx = start + i
                val sample = if (idx < samples.size) samples[idx] else 0.0f
                sample * hannWindow[i]
            }
            computePowerSpectrum(frame)
        }

        // 3. Apply mel filterbank: [80][201] × [nFrames][201] → [80][nFrames]
        val melSpec = Array(N_MELS) { m ->
            FloatArray(nFrames) { t ->
                var sum = 0.0f
                for (k in 0 until N_FREQ_BINS) {
                    sum += melFilterbank[m][k] * spectrogram[t][k]
                }
                sum
            }
        }

        // 4. Log scale + normalize
        // Find max value for dynamic range compression
        var maxVal = Float.MIN_VALUE
        for (m in 0 until N_MELS) {
            for (t in 0 until nFrames) {
                if (melSpec[m][t] > maxVal) maxVal = melSpec[m][t]
            }
        }

        val logMax = if (maxVal > 0) log10(maxVal) else 0.0f
        val floor = logMax - 8.0f  // 80 dB dynamic range

        for (m in 0 until N_MELS) {
            for (t in 0 until nFrames) {
                val logVal = log10(melSpec[m][t].coerceAtLeast(1e-10f))
                val clamped = max(logVal, floor)
                melSpec[m][t] = (clamped + 4.0f) / 4.0f  // Normalize to ~[-1, 1]
            }
        }

        // 5. Pad or truncate to exactly N_FRAMES (3000)
        val result = Array(1) {
            Array(N_MELS) { m ->
                FloatArray(N_FRAMES) { t ->
                    if (t < nFrames) melSpec[m][t] else 0.0f
                }
            }
        }

        Log.d(TAG, "Mel spectrogram: ${nFrames} frames → [1, $N_MELS, $N_FRAMES]")
        return result
    }

    /**
     * Compute power spectrum of a single frame using direct DFT.
     * Returns magnitude squared for positive frequencies (0..200).
     *
     * Uses direct DFT (O(n²)) since n_fft=400 is not a power of 2.
     * With precomputed twiddle factors, this runs in ~80K ops per frame,
     * ~240M ops total for 30s of audio — acceptable for on-device.
     */
    private fun computePowerSpectrum(frame: FloatArray): FloatArray {
        val power = FloatArray(N_FREQ_BINS)
        for (k in 0 until N_FREQ_BINS) {
            var re = 0.0f
            var im = 0.0f
            val cosRow = cosTable[k]
            val sinRow = sinTable[k]
            for (n in 0 until N_FFT) {
                re += frame[n] * cosRow[n]
                im += frame[n] * sinRow[n]
            }
            power[k] = re * re + im * im
        }
        return power
    }

    /**
     * Build 80-band mel filterbank matrix [80][201].
     * Uses HTK mel scale: mel = 2595 * log10(1 + f/700)
     * Triangle filters equally spaced in mel frequency domain.
     */
    private fun buildMelFilterbank(): Array<FloatArray> {
        val melMin = hzToMel(F_MIN)
        val melMax = hzToMel(F_MAX)

        // Center frequencies in Hz for each mel bin (+2 for edges)
        val melPoints = FloatArray(N_MELS + 2) { i ->
            melMin + i * (melMax - melMin) / (N_MELS + 1)
        }
        val hzPoints = FloatArray(melPoints.size) { melToHz(melPoints[it]) }

        // Convert Hz to FFT bin indices
        // bin = floor((n_fft + 1) * freq / sample_rate)
        val binPoints = IntArray(hzPoints.size) { i ->
            floor((N_FFT + 1).toFloat() * hzPoints[i] / SAMPLE_RATE).toInt()
                .coerceIn(0, N_FFT / 2)
        }

        // Build filterbank
        val filterbank = Array(N_MELS) { FloatArray(N_FREQ_BINS) }
        for (m in 0 until N_MELS) {
            val left = binPoints[m]
            val center = binPoints[m + 1]
            val right = binPoints[m + 2]

            // Rising slope
            for (k in left until center) {
                if (center != left) {
                    filterbank[m][k] = (k - left).toFloat() / (center - left)
                }
            }
            // Falling slope
            for (k in center until right) {
                if (right != center) {
                    filterbank[m][k] = (right - k).toFloat() / (right - center)
                }
            }
        }

        return filterbank
    }

    /** HTK mel scale: mel = 2595 * log10(1 + f/700) */
    private fun hzToMel(hz: Float): Float = 2595.0f * log10(1.0f + hz / 700.0f)

    /** Inverse HTK mel scale: f = 700 * (10^(mel/2595) - 1) */
    private fun melToHz(mel: Float): Float = 700.0f * (10.0f.pow(mel / 2595.0f) - 1.0f)
}
