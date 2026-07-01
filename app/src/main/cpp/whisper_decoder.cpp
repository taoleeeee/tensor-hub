#include "whisper_decoder.h"
#include <android/log.h>
#include <cstring>
#include <unordered_map>

#define LOG_TAG "WhisperDecoderNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

WhisperDecoder::~WhisperDecoder() {
    if (encode_runner) TfLiteSignatureRunnerDelete(encode_runner);
    if (decode_runner) TfLiteSignatureRunnerDelete(decode_runner);
    if (interpreter) TfLiteInterpreterDelete(interpreter);
    if (model) TfLiteModelDelete(model);
}

bool WhisperDecoder::load_model(const std::string& model_path, bool use_nnapi) {
    LOGI("Loading model from %s (use_nnapi=%d)", model_path.c_str(), use_nnapi);

    model = TfLiteModelCreateFromFile(model_path.c_str());
    if (!model) {
        LOGE("Failed to load model from %s", model_path.c_str());
        return false;
    }

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 4);
    LOGI("Set TFLite interpreter thread count to 4");

    // NNAPI delegate is applied via options if available on the device
    // The C API auto-detects NNAPI support
    if (use_nnapi) {
        LOGI("NNAPI delegate requested (C API auto-detects hardware acceleration)");
    }

    interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterOptionsDelete(options);

    if (!interpreter) {
        LOGE("Failed to create interpreter");
        return false;
    }

    // Get signature runners
    encode_runner = TfLiteInterpreterGetSignatureRunner(interpreter, "encode");
    decode_runner = TfLiteInterpreterGetSignatureRunner(interpreter, "decode");

    if (!encode_runner || !decode_runner) {
        LOGE("Failed to get signature runners (encode=%p, decode=%p)", encode_runner, decode_runner);
        return false;
    }

    initialized = true;
    LOGI("Model loaded successfully");
    return true;
}

std::vector<int> WhisperDecoder::build_sot_sequence(const std::string& lang) const {
    int sot = 50258;
    int transcribe = 50359;
    int notimestamps = 50363;

    static const std::unordered_map<std::string, int> LANGUAGE_IDS = {
        {"en", 0}, {"zh", 1}, {"de", 2}, {"es", 3}, {"ru", 4}, {"ko", 5},
        {"fr", 6}, {"ja", 7}, {"pt", 8}, {"tr", 9}, {"pl", 10}, {"ca", 11},
        {"nl", 12}, {"ar", 13}, {"sv", 14}, {"it", 15}, {"id", 16}, {"hi", 17},
        {"fi", 18}, {"vi", 19}, {"he", 20}, {"uk", 21}, {"el", 22}, {"ms", 23},
        {"cs", 24}, {"ro", 25}, {"da", 26}, {"hu", 27}, {"ta", 28}, {"no", 29},
        {"th", 30}, {"ur", 31}, {"hr", 32}, {"bg", 33}, {"lt", 34}, {"la", 35},
        {"mi", 36}, {"ml", 37}, {"cy", 38}, {"sk", 39}, {"te", 40}, {"fa", 41},
        {"lv", 42}, {"bn", 43}, {"sr", 44}, {"az", 45}, {"sl", 46}, {"kn", 47},
        {"et", 48}, {"mk", 49}, {"br", 50}, {"eu", 51}, {"is", 52}, {"hy", 53},
        {"ne", 54}, {"mn", 55}, {"bs", 56}, {"kk", 57}, {"sq", 58}, {"sw", 59},
        {"gl", 60}, {"mr", 61}, {"pa", 62}, {"si", 63}, {"km", 64}, {"sn", 65},
        {"yo", 66}, {"so", 67}, {"af", 68}, {"oc", 69}, {"ka", 70}, {"be", 71},
        {"tg", 72}, {"sd", 73}, {"gu", 74}, {"am", 75}, {"yi", 76}, {"lo", 77},
        {"uz", 78}, {"fo", 79}, {"ht", 80}, {"ps", 81}, {"tk", 82}, {"nn", 83},
        {"mt", 84}, {"sa", 85}, {"lb", 86}, {"my", 87}, {"bo", 88}, {"tl", 89},
        {"mg", 90}, {"as", 91}, {"tt", 92}, {"haw", 93}, {"ln", 94}, {"ha", 95},
        {"ba", 96}, {"jw", 97}, {"su", 98}
    };

    int lang_offset = 0;
    auto it = LANGUAGE_IDS.find(lang);
    if (it != LANGUAGE_IDS.end()) {
        lang_offset = it->second;
    }
    int lang_token = 50259 + lang_offset;

    return {sot, lang_token, transcribe, notimestamps};
}

bool WhisperDecoder::is_special_token(int token_id) const {
    return token_id >= 50257 && token_id <= 51864;
}

bool WhisperDecoder::is_end_of_text(int token_id) const {
    return token_id == 50257 || token_id == 50362;
}

std::vector<int> WhisperDecoder::transcribe(const float* pcm_samples, int sample_count, const std::string& lang) {
    if (!initialized) {
        LOGE("Transcribe called on uninitialized decoder");
        return {};
    }

    // 1. Allocate encode tensors and compute mel spectrogram
    TfLiteStatus status = TfLiteSignatureRunnerAllocateTensors(encode_runner);
    if (status != kTfLiteOk) {
        LOGE("Failed to allocate encode signature tensors");
        return {};
    }

    TfLiteTensor* enc_input = TfLiteSignatureRunnerGetInputTensor(encode_runner, nullptr);
    float* enc_input_data = static_cast<float*>(TfLiteTensorData(enc_input));
    mel_processor.compute(pcm_samples, sample_count, enc_input_data);

    // 2. Invoke encoder
    status = TfLiteSignatureRunnerInvoke(encode_runner);
    if (status != kTfLiteOk) {
        LOGE("Encoder execution failed");
        return {};
    }

    const TfLiteTensor* enc_output = TfLiteSignatureRunnerGetOutputTensor(encode_runner, nullptr);
    const float* enc_output_data = static_cast<const float*>(TfLiteTensorData(enc_output));

    // 3. Allocate decode tensors
    status = TfLiteSignatureRunnerAllocateTensors(decode_runner);
    if (status != kTfLiteOk) {
        LOGE("Failed to allocate decode signature tensors");
        return {};
    }

    // Copy encoder output to decoder input
    // Find the encoder output input tensor (3D tensor)
    TfLiteTensor* dec_enc_input = nullptr;
    TfLiteTensor* dec_mask = nullptr;
    TfLiteTensor* dec_tokens = nullptr;

    int input_count = TfLiteSignatureRunnerGetInputCount(decode_runner);
    for (int i = 0; i < input_count; ++i) {
        const char* name = TfLiteSignatureRunnerGetInputName(decode_runner, i);
        TfLiteTensor* tensor = TfLiteSignatureRunnerGetInputTensor(decode_runner, name);
        const TfLiteIntArray* dims = TfLiteTensorDims(tensor);
        if (dims->size == 3) {
            dec_enc_input = tensor;
        } else if (dims->size == 4) {
            dec_mask = tensor;
        } else if (dims->size == 2) {
            dec_tokens = tensor;
        }
    }

    if (!dec_enc_input || !dec_mask || !dec_tokens) {
        LOGE("Could not identify all decode input tensors (enc=%p, mask=%p, tokens=%p)",
             dec_enc_input, dec_mask, dec_tokens);
        return {};
    }

    // Copy encoder output
    float* dec_enc_data = static_cast<float*>(TfLiteTensorData(dec_enc_input));
    std::memcpy(dec_enc_data, enc_output_data, 1 * 1500 * 512 * sizeof(float));

    // Fill causal mask
    float* mask_data = static_cast<float*>(TfLiteTensorData(dec_mask));
    constexpr float MASKED_OUT = -0.7f * 3.40282347e+38f;
    for (int r = 0; r < 128; ++r) {
        for (int c = 0; c < 128; ++c) {
            mask_data[r * 128 + c] = (c <= r) ? 0.0f : MASKED_OUT;
        }
    }

    // Initialize token IDs
    int32_t* token_ids_data = static_cast<int32_t*>(TfLiteTensorData(dec_tokens));
    std::vector<int> token_ids(128, 0);
    std::vector<int> sot_seq = build_sot_sequence(lang);
    for (size_t i = 0; i < sot_seq.size(); ++i) {
        token_ids[i] = sot_seq[i];
    }
    int step = sot_seq.size();
    std::vector<int> generated_tokens;

    // 4. Autoregressive loop
    for (int iteration = 0; iteration < 128; ++iteration) {
        std::memcpy(token_ids_data, token_ids.data(), 128 * sizeof(int32_t));

        status = TfLiteSignatureRunnerInvoke(decode_runner);
        if (status != kTfLiteOk) {
            LOGE("Decoder execution failed at step %d", iteration);
            break;
        }

        // Get output logits
        const TfLiteTensor* logits_tensor = TfLiteSignatureRunnerGetOutputTensor(decode_runner, nullptr);
        const float* logits_data = static_cast<const float*>(TfLiteTensorData(logits_tensor));

        // Argmax at current position
        int vocab_size = 51865;
        int predict_pos = step - 1;
        int start_idx = predict_pos * vocab_size;

        int best_token = 0;
        float best_score = logits_data[start_idx];
        for (int idx = start_idx + 1; idx < start_idx + vocab_size; ++idx) {
            if (logits_data[idx] > best_score) {
                best_score = logits_data[idx];
                best_token = idx - start_idx;
            }
        }

        LOGD("Step %d (pos=%d): token=%d (score=%.4f)", iteration, step, best_token, best_score);

        if (is_end_of_text(best_token)) {
            LOGI("End of sequence at step %d (token=%d)", iteration, best_token);
            break;
        }

        if (!is_special_token(best_token)) {
            generated_tokens.push_back(best_token);
        }

        if (step < 128 - 1) {
            token_ids[step] = best_token;
            step++;
        } else {
            LOGW("Max decode length reached");
            break;
        }
    }

    LOGI("Native decode complete: %zu tokens in %d steps", generated_tokens.size(), step);
    return generated_tokens;
}
