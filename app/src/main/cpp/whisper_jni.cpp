#include <jni.h>
#include <string>
#include <vector>
#include "whisper_decoder.h"
#include <android/log.h>

#define LOG_TAG "WhisperJNINative"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_taoleeeee_tensorhub_inference_WhisperNative_initNative(
        JNIEnv* env,
        jobject thiz,
        jstring model_path_obj,
        jboolean use_nnapi) {
    if (!model_path_obj) {
        LOGE("initNative: modelPath is null");
        return 0;
    }
    
    const char* model_path = env->GetStringUTFChars(model_path_obj, nullptr);
    if (!model_path) {
        return 0;
    }
    
    WhisperDecoder* decoder = new WhisperDecoder();
    bool success = decoder->load_model(model_path, use_nnapi);
    
    env->ReleaseStringUTFChars(model_path_obj, model_path);
    
    if (!success) {
        delete decoder;
        return 0;
    }
    
    return reinterpret_cast<jlong>(decoder);
}

JNIEXPORT jintArray JNICALL
Java_com_taoleeeee_tensorhub_inference_WhisperNative_transcribeNative(
        JNIEnv* env,
        jobject thiz,
        jlong handle,
        jfloatArray pcm_samples_obj,
        jstring lang_obj) {
        
    WhisperDecoder* decoder = reinterpret_cast<WhisperDecoder*>(handle);
    if (!decoder) {
        LOGE("transcribeNative: invalid native decoder handle");
        return nullptr;
    }
    
    if (!pcm_samples_obj) {
        LOGE("transcribeNative: pcmSamples array is null");
        return nullptr;
    }
    
    jsize sample_count = env->GetArrayLength(pcm_samples_obj);
    jfloat* samples_ptr = env->GetFloatArrayElements(pcm_samples_obj, nullptr);
    if (!samples_ptr) {
        return nullptr;
    }
    
    std::string lang = "en";
    if (lang_obj) {
        const char* lang_str = env->GetStringUTFChars(lang_obj, nullptr);
        if (lang_str) {
            lang = lang_str;
            env->ReleaseStringUTFChars(lang_obj, lang_str);
        }
    }
    
    // Run native transcription
    std::vector<int> tokens = decoder->transcribe(samples_ptr, sample_count, lang);
    
    // Release native float array
    env->ReleaseFloatArrayElements(pcm_samples_obj, samples_ptr, JNI_ABORT);
    
    // Copy output tokens to Java jintArray
    jintArray result = env->NewIntArray(tokens.size());
    if (result) {
        env->SetIntArrayRegion(result, 0, tokens.size(), reinterpret_cast<const jint*>(tokens.data()));
    }
    
    return result;
}

JNIEXPORT void JNICALL
Java_com_taoleeeee_tensorhub_inference_WhisperNative_releaseNative(
        JNIEnv* env,
        jobject thiz,
        jlong handle) {
    WhisperDecoder* decoder = reinterpret_cast<WhisperDecoder*>(handle);
    if (decoder) {
        delete decoder;
    }
}

}
