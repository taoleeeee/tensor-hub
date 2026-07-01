// Minimal TFLite C API declarations for Whisper native build.
// Avoids transitive header deps from the full c_api.h.
// Linking against libtensorflowlite_jni.so provides the actual symbols.

#ifndef TFLITE_MINIMAL_H
#define TFLITE_MINIMAL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Status codes
typedef enum TfLiteStatus {
    kTfLiteOk = 0,
    kTfLiteError = 1,
    kTfLiteDelegateError = 2,
    kTfLiteDelegateDataNotFound = 3,
    kTfLiteDelegateDataError = 4,
    kTfLiteDelegateErrorMissingOutput = 5,
    kTfLiteDelegateErrorMissingInput = 6,
} TfLiteStatus;

// Opaque types
typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreter TfLiteInterpreter;
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;
typedef struct TfLiteSignatureRunner TfLiteSignatureRunner;
typedef struct TfLiteTensor TfLiteTensor;

// Array of ints (for tensor dimensions)
typedef struct TfLiteIntArray {
    int size;
    int data[];  // flexible array member
} TfLiteIntArray;

// Model
TfLiteModel* TfLiteModelCreateFromFile(const char* model_path);
void TfLiteModelDelete(TfLiteModel* model);

// Interpreter options
TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate();
void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions* options);
void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions* options, int32_t num_threads);

// Interpreter
TfLiteInterpreter* TfLiteInterpreterCreate(const TfLiteModel* model, const TfLiteInterpreterOptions* options);
void TfLiteInterpreterDelete(TfLiteInterpreter* interpreter);
TfLiteStatus TfLiteInterpreterAllocateTensors(TfLiteInterpreter* interpreter);

// Signature runners
TfLiteSignatureRunner* TfLiteInterpreterGetSignatureRunner(TfLiteInterpreter* interpreter, const char* signature_key);
void TfLiteSignatureRunnerDelete(TfLiteSignatureRunner* runner);
TfLiteStatus TfLiteSignatureRunnerAllocateTensors(TfLiteSignatureRunner* runner);
TfLiteStatus TfLiteSignatureRunnerInvoke(TfLiteSignatureRunner* runner);

// Signature runner I/O
size_t TfLiteSignatureRunnerGetInputCount(const TfLiteSignatureRunner* runner);
size_t TfLiteSignatureRunnerGetOutputCount(const TfLiteSignatureRunner* runner);
const char* TfLiteSignatureRunnerGetInputName(const TfLiteSignatureRunner* runner, size_t index);
const char* TfLiteSignatureRunnerGetOutputName(const TfLiteSignatureRunner* runner, size_t index);
TfLiteTensor* TfLiteSignatureRunnerGetInputTensor(const TfLiteSignatureRunner* runner, const char* input_name);
TfLiteTensor* TfLiteSignatureRunnerGetOutputTensor(const TfLiteSignatureRunner* runner, const char* output_name);

// Tensor access
int TfLiteTensorNumDims(const TfLiteTensor* tensor);
int TfLiteTensorDim(const TfLiteTensor* tensor, int dim_index);
void* TfLiteTensorData(const TfLiteTensor* tensor);

#ifdef __cplusplus
}
#endif

#endif // TFLITE_MINIMAL_H
