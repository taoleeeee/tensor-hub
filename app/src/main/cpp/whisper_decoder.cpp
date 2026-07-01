#include "whisper_decoder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include <android/log.h>
#include <cstring>
#include <unordered_map>

#define LOG_TAG "WhisperDecoderNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

bool WhisperDecoder::load_model(const std::string& model_path, bool use_nnapi) {
    LOGI("Loading model from %s (use_nnapi=%d)", model_path.c_str(), use_nnapi);
    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        LOGE("Failed to build FlatBufferModel from %s", model_path.c_str());
        return false;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&interpreter);
    if (!interpreter) {
        LOGE("Failed to build interpreter");
        return false;
    }

    // Set threads to 4 to maximize CPU utilization on Cortex big/mid cores
    interpreter->SetNumThreads(4);
    LOGI("Set TFLite interpreter thread count to 4");

    if (use_nnapi) {
        LOGI("Applying native NNAPI delegate to the interpreter...");
        tflite::StatefulNnApiDelegate::Options delegate_options;
        auto nnapi_delegate = std::make_unique<tflite::StatefulNnApiDelegate>(delegate_options);
        if (interpreter->ModifyGraphWithDelegate(std::move(nnapi_delegate)) != kTfLiteOk) {
            LOGW("Failed to apply native NNAPI delegate, falling back to CPU");
        } else {
            LOGI("Native NNAPI delegate applied successfully");
        }
    }

    // Allocate tensors initially
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOGE("Failed to allocate interpreter tensors");
        return false;
    }

    encode_runner = interpreter->GetSignatureRunner("encode");
    decode_runner = interpreter->GetSignatureRunner("decode");

    if (!encode_runner || !decode_runner) {
        LOGE("Failed to get signature runners (encode=%p, decode=%p)", encode_runner, decode_runner);
        return false;
    }

    if (!resolve_signature_names()) {
        LOGE("Failed to resolve signature names");
        return false;
    }

    initialized = true;
    LOGI("Model loaded successfully");
    return true;
}

bool WhisperDecoder::resolve_signature_names() {
    if (encode_runner->input_size() == 0 || encode_runner->output_size() == 0) {
        LOGE("Encode signature has empty inputs/outputs");
        return false;
    }
    encode_input_name = encode_runner->input_names()[0];
    encode_output_name = encode_runner->output_names()[0];

    auto input_names = decode_runner->input_names();
    for (size_t i = 0; i < decode_runner->input_size(); ++i) {
        const char* name = input_names[i];
        TfLiteTensor* tensor = decode_runner->input_tensor(name);
        if (!tensor) continue;
        int num_dims = tensor->dims->size;
        if (num_dims == 3) {
            decode_enc_output_name = name;
        } else if (num_dims == 2) {
            decode_token_ids_name = name;
        } else if (num_dims == 4) {
            decode_mask_name = name;
        }
    }

    if (decode_runner->output_size() > 0) {
        decode_logits_name = decode_runner->output_names()[0];
    }

    LOGI("Resolved encode: in='%s', out='%s'", encode_input_name.c_str(), encode_output_name.c_str());
    LOGI("Resolved decode: enc_out='%s', tokens='%s', mask='%s', logits='%s'",
         decode_enc_output_name.c_str(), decode_token_ids_name.c_str(), decode_mask_name.c_str(), decode_logits_name.c_str());

    return !decode_enc_output_name.empty() && !decode_token_ids_name.empty() && 
           !decode_mask_name.empty() && !decode_logits_name.empty();
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

    // 1. Allocate encode inputs & compute mel spectrogram directly into TFLite memory
    if (encode_runner->AllocateTensors() != kTfLiteOk) {
        LOGE("Failed to allocate encode signature tensors");
        return {};
    }
    TfLiteTensor* enc_input_tensor = encode_runner->input_tensor(encode_input_name.c_str());
    float* enc_input_data = enc_input_tensor->data.f;
    mel_processor.compute(pcm_samples, sample_count, enc_input_data);

    // 2. Invoke encoder
    if (encode_runner->Invoke() != kTfLiteOk) {
        LOGE("Encoder execution failed");
        return {};
    }
    const TfLiteTensor* enc_output_tensor = encode_runner->output_tensor(encode_output_name.c_str());
    const float* enc_output_data = enc_output_tensor->data.f;

    // 3. Allocate decode inputs & initialize buffers
    if (decode_runner->AllocateTensors() != kTfLiteOk) {
        LOGE("Failed to allocate decode signature tensors");
        return {};
    }

    // Copy encoder output directly to decoder input
    TfLiteTensor* dec_enc_input_tensor = decode_runner->input_tensor(decode_enc_output_name.c_str());
    std::memcpy(dec_enc_input_tensor->data.f, enc_output_data, 1 * 1500 * 512 * sizeof(float));

    // Fill causal mask: lower triangular is 0.0, rest is -0.7 * Float.MAX_VALUE (approx -2.4e38f)
    TfLiteTensor* dec_mask_tensor = decode_runner->input_tensor(decode_mask_name.c_str());
    float* mask_data = dec_mask_tensor->data.f;
    constexpr float MASKED_IN = 0.0f;
    constexpr float MASKED_OUT = -0.7f * 3.40282347e+38f;
    for (int r = 0; r < 128; ++r) {
        for (int c = 0; c < 128; ++c) {
            mask_data[r * 128 + c] = (c <= r) ? MASKED_IN : MASKED_OUT;
        }
    }

    // Initialize token IDs
    TfLiteTensor* dec_tokens_tensor = decode_runner->input_tensor(decode_token_ids_name.c_str());
    int32_t* token_ids_data = dec_tokens_tensor->data.i32;

    std::vector<int> token_ids(128, 0);
    std::vector<int> sot_seq = build_sot_sequence(lang);
    for (size_t i = 0; i < sot_seq.size(); ++i) {
        token_ids[i] = sot_seq[i];
    }
    int step = sot_seq.size();
    std::vector<int> generated_tokens;

    // 4. Autoregressive Loop (natively in C++)
    for (int iteration = 0; iteration < 128; ++iteration) {
        // Copy current token ids to decoder input
        std::memcpy(token_ids_data, token_ids.data(), 128 * sizeof(int32_t));

        if (decode_runner->Invoke() != kTfLiteOk) {
            LOGE("Decoder execution failed at step %d", iteration);
            break;
        }

        // Get output logits
        const TfLiteTensor* logits_tensor = decode_runner->output_tensor(decode_logits_name.c_str());
        const float* logits_data = logits_tensor->data.f;

        // Argmax at step - 1
        int predict_pos = step - 1;
        int vocab_size = 51865;
        int start_idx = predict_pos * vocab_size;
        int end_idx = (predict_pos + 1) * vocab_size;

        int best_token = 0;
        float best_score = logits_data[start_idx];
        for (int idx = start_idx + 1; idx < end_idx; ++idx) {
            if (logits_data[idx] > best_score) {
                best_score = logits_data[idx];
                best_token = idx - start_idx;
            }
        }

        LOGD("Step %d (pos=%d, readAt=%d): token=%d (score=%.4f)", iteration, step, predict_pos, best_token, best_score);

        if (is_end_of_text(best_token)) {
            LOGI("End of sequence detected at step %d (token=%d)", iteration, best_token);
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

    LOGI("Native decode complete: generated %zu tokens in %d steps", generated_tokens.size(), step);
    return generated_tokens;
}
