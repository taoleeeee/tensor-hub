package com.taoleeeee.tensorhub.inference

import android.util.Log
import java.io.File

/**
 * Kotlin JNI bridge for Whisper C++ native execution.
 * Handles loading `libwhisper_native.so` and manages the native decoder lifecycle.
 */
class WhisperNative(modelFile: File, useNnapi: Boolean = true) : AutoCloseable {

    companion object {
        private const val TAG = "WhisperNative"

        init {
            try {
                System.loadLibrary("whisper_native")
                Log.i(TAG, "Loaded native library whisper_native")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load native library whisper_native: ${e.message}", e)
            }
        }
    }

    private var nativeHandle: Long = 0

    init {
        nativeHandle = initNative(modelFile.absolutePath, useNnapi)
        if (nativeHandle == 0L) {
            throw RuntimeException("Failed to initialize native Whisper decoder for model path: ${modelFile.absolutePath} (nnapi=$useNnapi)")
        }
        Log.i(TAG, "Initialized native decoder with handle $nativeHandle (nnapi=$useNnapi)")
    }

    /**
     * Transcribe 16kHz mono audio float samples using the native C++ pipeline.
     * Returns an array of predicted token IDs.
     */
    fun transcribe(pcmSamples: FloatArray, language: String): IntArray {
        if (nativeHandle == 0L) {
            throw IllegalStateException("Native decoder has been released")
        }
        return transcribeNative(nativeHandle, pcmSamples, language)
            ?: throw RuntimeException("Native transcription execution failed")
    }

    override fun close() {
        if (nativeHandle != 0L) {
            releaseNative(nativeHandle)
            Log.i(TAG, "Released native decoder handle $nativeHandle")
            nativeHandle = 0
        }
    }

    // Native JNI declarations
    private external fun initNative(modelPath: String, useNnapi: Boolean): Long
    private external fun transcribeNative(handle: Long, pcmSamples: FloatArray, language: String): IntArray?
    private external fun releaseNative(handle: Long)
}
