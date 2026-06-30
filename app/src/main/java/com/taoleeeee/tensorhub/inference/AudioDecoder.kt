package com.taoleeeee.tensorhub.inference

import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Decodes audio files (WAV) to raw 16kHz mono float samples.
 *
 * Whisper requires 16kHz mono audio as input. This decoder handles:
 * - WAV files (PCM 16-bit, PCM float32)
 * - Mono and stereo → mono conversion
 * - Resampling to 16kHz via linear interpolation
 */
object AudioDecoder {

    private const val TAG = "AudioDecoder"
    private const val TARGET_SAMPLE_RATE = 16000

    /**
     * Decode an audio file to 16kHz mono float samples.
     * Returns samples normalized to [-1, 1] range.
     */
    fun decode(file: File): Result<FloatArray> {
        return try {
        val raf = RandomAccessFile(file, "r")
        val header = ByteArray(44)
        raf.readFully(header)

        // Parse RIFF header
        val riff = String(header, 0, 4)
        if (riff != "RIFF") {
            raf.close()
            return Result.failure(Exception("Not a WAV file (missing RIFF header): $riff"))
        }
        val wave = String(header, 8, 4)
        if (wave != "WAVE") {
            raf.close()
            return Result.failure(Exception("Not a WAV file (missing WAVE marker): $wave"))
        }

        // Parse fmt chunk
        val audioFormat = (header[20].toInt() and 0xFF) or ((header[21].toInt() and 0xFF) shl 8)
        val numChannels = (header[22].toInt() and 0xFF) or ((header[23].toInt() and 0xFF) shl 8)
        val sampleRate = ((header[24].toInt() and 0xFF) or
                ((header[25].toInt() and 0xFF) shl 8) or
                ((header[26].toInt() and 0xFF) shl 16) or
                ((header[27].toInt() and 0xFF) shl 24))
        val bitsPerSample = (header[34].toInt() and 0xFF) or ((header[35].toInt() and 0xFF) shl 8)

        // Find data chunk (may not be at fixed offset if there are extra chunks)
        var dataOffset = 36L
        var dataSize = ((header[36].toInt() and 0xFF) or
                ((header[37].toInt() and 0xFF) shl 8) or
                ((header[38].toInt() and 0xFF) shl 16) or
                ((header[39].toInt() and 0xFF) shl 24)).toLong()

        // If fmt chunk has extra params, data chunk may be further out
        // Search for "data" marker
        val dataMarker = String(header, 36, 4)
        if (dataMarker != "data") {
            // Scan file for "data" chunk
            raf.seek(12)
            var found = false
            while (raf.filePointer < raf.length() - 8) {
                val chunkId = ByteArray(4)
                raf.readFully(chunkId)
                val chunkIdStr = String(chunkId)
                val chunkSizeBuf = ByteArray(4)
                raf.readFully(chunkSizeBuf)
                val chunkSize = ((chunkSizeBuf[0].toInt() and 0xFF) or
                        ((chunkSizeBuf[1].toInt() and 0xFF) shl 8) or
                        ((chunkSizeBuf[2].toInt() and 0xFF) shl 16) or
                        ((chunkSizeBuf[3].toInt() and 0xFF) shl 24)).toLong()
                if (chunkIdStr == "data") {
                    dataOffset = raf.filePointer
                    dataSize = chunkSize
                    found = true
                    break
                }
                raf.seek(raf.filePointer + chunkSize)
            }
            if (!found) {
                raf.close()
                return Result.failure(Exception("Could not find data chunk in WAV file"))
            }
        } else {
            dataOffset = 44
        }

        Log.i(TAG, "WAV: ${numChannels}ch, ${sampleRate}Hz, ${bitsPerSample}bit, " +
                "format=$audioFormat, dataSize=$dataSize")

        // Read raw samples
        raf.seek(dataOffset)
        val rawData = ByteArray(dataSize.toInt().coerceAtMost((raf.length() - dataOffset).toInt()))
        raf.readFully(rawData)
        raf.close()

        // Convert to mono float samples
        val samples = when {
            audioFormat == 1 && bitsPerSample == 16 -> decodePcm16(rawData, numChannels)
            audioFormat == 1 && bitsPerSample == 24 -> decodePcm24(rawData, numChannels)
            audioFormat == 3 && bitsPerSample == 32 -> decodeFloat32(rawData, numChannels)
            audioFormat == 1 && bitsPerSample == 8 -> decodePcm8(rawData, numChannels)
            else -> return Result.failure(
                Exception("Unsupported WAV format: format=$audioFormat, bits=$bitsPerSample")
            )
        }

        // Resample to 16kHz if needed
        val resampled = if (sampleRate != TARGET_SAMPLE_RATE) {
            Log.i(TAG, "Resampling ${sampleRate}Hz → ${TARGET_SAMPLE_RATE}Hz")
            resample(samples, sampleRate, TARGET_SAMPLE_RATE)
        } else {
            samples
        }

        Log.i(TAG, "Decoded ${resampled.size} samples at ${TARGET_SAMPLE_RATE}Hz " +
                "(${resampled.size.toFloat() / TARGET_SAMPLE_RATE}s)")

        Result.success(resampled)
    } catch (e: Exception) {
        Log.e(TAG, "Audio decode failed: ${e.message}", e)
        Result.failure(e)
        }
    }

    /**
     * Decode 16-bit PCM to float [-1, 1], converting to mono.
     */
    private fun decodePcm16(data: ByteArray, channels: Int): FloatArray {
        val totalSamples = data.size / 2
        val monoSamples = totalSamples / channels
        val result = FloatArray(monoSamples)
        val bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)

        for (i in 0 until monoSamples) {
            var sum = 0.0f
            for (ch in 0 until channels) {
                sum += bb.short / 32768.0f
            }
            result[i] = sum / channels
        }
        return result
    }

    /**
     * Decode 24-bit PCM to float [-1, 1], converting to mono.
     */
    private fun decodePcm24(data: ByteArray, channels: Int): FloatArray {
        val bytesPerSample = 3
        val totalSamples = data.size / (bytesPerSample * channels)
        val result = FloatArray(totalSamples)
        var offset = 0

        for (i in 0 until totalSamples) {
            var sum = 0.0f
            for (ch in 0 until channels) {
                val low = data[offset].toInt() and 0xFF
                val mid = data[offset + 1].toInt() and 0xFF
                val high = data[offset + 2].toInt()
                var value = low or (mid shl 8) or (high shl 16)
                if (value and 0x800000 != 0) value = value or -0x1000000  // sign extend
                sum += value / 8388608.0f
                offset += bytesPerSample
            }
            result[i] = sum / channels
        }
        return result
    }

    /**
     * Decode float32 PCM to float, converting to mono.
     */
    private fun decodeFloat32(data: ByteArray, channels: Int): FloatArray {
        val totalSamples = data.size / 4
        val monoSamples = totalSamples / channels
        val result = FloatArray(monoSamples)
        val bb = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)

        for (i in 0 until monoSamples) {
            var sum = 0.0f
            for (ch in 0 until channels) {
                sum += bb.float
            }
            result[i] = sum / channels
        }
        return result
    }

    /**
     * Decode 8-bit unsigned PCM to float [-1, 1], converting to mono.
     */
    private fun decodePcm8(data: ByteArray, channels: Int): FloatArray {
        val totalSamples = data.size / channels
        val result = FloatArray(totalSamples)
        var offset = 0

        for (i in 0 until totalSamples) {
            var sum = 0.0f
            for (ch in 0 until channels) {
                sum += ((data[offset].toInt() and 0xFF) - 128) / 128.0f
                offset++
            }
            result[i] = sum / channels
        }
        return result
    }

    /**
     * Resample audio using linear interpolation.
     */
    private fun resample(samples: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        if (fromRate == toRate) return samples
        val ratio = fromRate.toDouble() / toRate
        val newLength = (samples.size / ratio).toInt()
        val result = FloatArray(newLength)
        for (i in 0 until newLength) {
            val srcPos = i * ratio
            val srcIdx = srcPos.toInt()
            val frac = (srcPos - srcIdx).toFloat()
            result[i] = if (srcIdx + 1 < samples.size) {
                samples[srcIdx] * (1 - frac) + samples[srcIdx + 1] * frac
            } else {
                samples[srcIdx]
            }
        }
        return result
    }
}
