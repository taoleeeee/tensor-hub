#ifndef TENSORHUB_WHISPER_DECODER_H
#define TENSORHUB_WHISPER_DECODER_H

#include <string>
#include <vector>
#include <memory>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/signature_runner.h"
#include "whisper_mel.h"

class WhisperDecoder {
private:
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    
    tflite::SignatureRunner* encode_runner = nullptr;
    tflite::SignatureRunner* decode_runner = nullptr;

    std::string encode_input_name;
    std::string encode_output_name;

    std::string decode_enc_output_name;
    std::string decode_token_ids_name;
    std::string decode_mask_name;
    std::string decode_logits_name;

    WhisperMel mel_processor;

    bool initialized = false;

    bool resolve_signature_names();
    std::vector<int> build_sot_sequence(const std::string& lang) const;
    bool is_special_token(int token_id) const;
    bool is_end_of_text(int token_id) const;

public:
    WhisperDecoder() = default;
    
    bool load_model(const std::string& model_path, bool use_nnapi = true);
    std::vector<int> transcribe(const float* pcm_samples, int sample_count, const std::string& lang);
};

#endif // TENSORHUB_WHISPER_DECODER_H
