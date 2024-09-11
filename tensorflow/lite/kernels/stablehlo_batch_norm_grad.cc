#include <iostream>

#include "kernel_util.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/add.h"
#include "tensorflow/lite/kernels/internal/reference/broadcast_to.h"
#include "tensorflow/lite/kernels/internal/reference/div.h"
#include "tensorflow/lite/kernels/internal/reference/mul.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/reference/sub.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/dequantize.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_batch_norm_grad {
namespace {

constexpr int kMaxTemporaryTensors = 21;
constexpr int32_t kMaxReduceRank = 6;
struct OpData {
 public:
  enum {
    kOperandTensor,
    kScaleTensor,
    kMeanTensor,
    kVarianceTensor,
    kGradOutputTensor
  };
  enum {
    kOutputGradOperandTensor,
    kOutputGradScaleTensor,
    kOutputGradOffsetTensor
  };
  int scratch_tensor_index;
};

void* Init(TfLiteContext* context, const char* options, size_t options_len) {
  OpData* data = new OpData;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

template <typename DataType>
TfLiteStatus ComputeSum(TfLiteContext* context, TfLiteNode* node,
                        const TfLiteTensor* operand, int feature_index,
                        TfLiteTensor* batch_sum) {
  int operand_rank = operand->dims->size;
  std::vector<int> dimarray;
  for (int i = 0; i < operand_rank; i++) {
    if (i != feature_index) {
      dimarray.push_back(i);
    }
  }
  int resolved_axis[kMaxReduceRank];
  int temp_index[kMaxReduceRank];
  TF_LITE_ENSURE(context,
                 reference_ops::ReduceGeneric<DataType>(
                     GetTensorData<DataType>(operand), operand->dims->data,
                     operand->dims->size, GetTensorData<DataType>(batch_sum),
                     batch_sum->dims->data, batch_sum->dims->size,
                     dimarray.data(), dimarray.size(), false, temp_index,
                     resolved_axis, static_cast<DataType>(0),
                     [](const DataType current, const DataType in) -> DataType {
                       return in + current;
                     }));
                       DataType* batch_sum_buffer =
      GetTensorData<DataType>(batch_sum);
                     for(int i=0;i<NumElements(batch_sum);++i){
                      std::cout<<"batch sum i "<<i<<"-"<<batch_sum_buffer[i]<<std::endl;
                     }

  return kTfLiteOk;
}

template <typename T>
void Sqrt(const T* input_data, T* output_data, int num_elements) {
  for (int i = 0; i < num_elements; ++i) {
    output_data[i] = std::sqrt(input_data[i]);
  }
}

TfLiteStatus PrepareTemporaries(TfLiteContext* context, TfLiteNode* node,
                                const TfLiteBatchNormGradParams* params,
                                const TfLiteTensor* operand,
                                const TfLiteTensor* grad_output,const TfLiteTensor* scale) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  context->AddTensors(context, kMaxTemporaryTensors,
                      &data->scratch_tensor_index);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(kMaxTemporaryTensors);

  // Set up epsilon_tensor
  node->temporaries->data[0] = data->scratch_tensor_index;
  TfLiteTensor* epsilon_tensor;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/0, &epsilon_tensor));
  TfLiteIntArray* epsilon_tensor_shape = TfLiteIntArrayCreate(1);
  epsilon_tensor_shape->data[0] = 1;
  epsilon_tensor->type = operand->type;
  epsilon_tensor->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, epsilon_tensor,
                                                   epsilon_tensor_shape));

  /* centered operand bcast*/
  node->temporaries->data[1] = data->scratch_tensor_index + 1;
  TfLiteTensor* centered_operand;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/1, &centered_operand));
  TfLiteIntArray* centered_operand_bcast_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    centered_operand_bcast_shape->data[i] = operand->dims->data[i];
  }
  centered_operand->type = operand->type;
  centered_operand->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, centered_operand,
                                          centered_operand_bcast_shape));

  /*stddev bcast*/
  node->temporaries->data[2] = data->scratch_tensor_index + 2;
  TfLiteTensor* stddev;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/2, &stddev));
  TfLiteIntArray* stddev_bcast_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    stddev_bcast_shape->data[i] = operand->dims->data[i];
  }
  stddev->type = operand->type;
  stddev->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, stddev, stddev_bcast_shape));

  // 1. Allocate and resize tensor for normalized_operand
  node->temporaries->data[3] = data->scratch_tensor_index + 3;
  TfLiteTensor* normalized_operand;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/3,
                                              &normalized_operand));
  TfLiteIntArray* normalized_operand_bcast_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    normalized_operand_bcast_shape->data[i] = operand->dims->data[i];
  }
  normalized_operand->type = operand->type;
  normalized_operand->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, normalized_operand,
                                          normalized_operand_bcast_shape));

  // 2. Allocate and resize tensor for elements_per_feature_tensor
  node->temporaries->data[4] = data->scratch_tensor_index + 4;
  TfLiteTensor* elements_per_feature_tensor;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/4,
                                              &elements_per_feature_tensor));
  TfLiteIntArray* elements_per_feature_tensor_shape = TfLiteIntArrayCreate(0);

  elements_per_feature_tensor->type = operand->type;
  elements_per_feature_tensor->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, elements_per_feature_tensor,
                                          elements_per_feature_tensor_shape));

  // 4. Allocate and resize tensor for i1
  node->temporaries->data[5] = data->scratch_tensor_index + 5;
  TfLiteTensor* i1;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/5, &i1));
  TfLiteIntArray* i1_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    i1_bcast_shape->data[i] = operand->dims->data[i];
  }
  i1->type = operand->type;
  i1->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, i1, i1_bcast_shape));

  // 7. Allocate and resize tensor for i4
  node->temporaries->data[6] = data->scratch_tensor_index + 6;
  TfLiteTensor* i4;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/6, &i4));
  TfLiteIntArray* i4_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    i4_bcast_shape->data[i] = operand->dims->data[i];
  }
  i4->type = operand->type;
  i4->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, i4, i4_bcast_shape));

  // 8. Allocate and resize tensor for i5
  node->temporaries->data[7] = data->scratch_tensor_index + 7;
  TfLiteTensor* i5;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/7, &i5));
  TfLiteIntArray* i5_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    i5_bcast_shape->data[i] = operand->dims->data[i];
  }
  i5->type = operand->type;
  i5->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, i5, i5_bcast_shape));

  // 9. Allocate and resize tensor for i6
  node->temporaries->data[8] = data->scratch_tensor_index + 8;
  TfLiteTensor* i6;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/8, &i6));
  TfLiteIntArray* i6_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    i6_bcast_shape->data[i] = operand->dims->data[i];
  }
  i6->type = operand->type;
  i6->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, i6, i6_bcast_shape));

  // 10. Allocate and resize tensor for grad_output_centered_operand_mu
  node->temporaries->data[9] = data->scratch_tensor_index + 9;
  TfLiteTensor* grad_output_centered_operand_mul;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/9,
                                     &grad_output_centered_operand_mul));
  TfLiteIntArray* grad_output_centered_operand_mul_bcast_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    grad_output_centered_operand_mul_bcast_shape->data[i] =
        operand->dims->data[i];
  }
  grad_output_centered_operand_mul->type = operand->type;
  grad_output_centered_operand_mul->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(
                                 context, grad_output_centered_operand_mul,
                                 grad_output_centered_operand_mul_bcast_shape));

  // 13. Allocate and resize tensor for grad_offset
  node->temporaries->data[10] = data->scratch_tensor_index + 10;
  TfLiteTensor* grad_output_reduced;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/10,
                                              &grad_output_reduced));
  TfLiteIntArray* grad_output_reduced_shape = TfLiteIntArrayCreate(1);
  grad_output_reduced_shape->data[0] =
      grad_output->dims->data[params->feature_index];

  grad_output_reduced->type = operand->type;
  grad_output_reduced->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, grad_output_reduced,
                                                   grad_output_reduced_shape));

  // 14. Allocate and resize tensor for i3_intermediate
  node->temporaries->data[11] = data->scratch_tensor_index + 11;
  TfLiteTensor* i3_intermediate;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/11, &i3_intermediate));
  TfLiteIntArray* i3_intermediate_shape = TfLiteIntArrayCreate(1);
  i3_intermediate_shape->data[0] =
      grad_output_centered_operand_mul->dims->data[params->feature_index];

  i3_intermediate->type = operand->type;
  i3_intermediate->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, i3_intermediate,
                                                   i3_intermediate_shape));

  // 14. Allocate and resize tensor for grad_Scale_intermediate
  node->temporaries->data[12] = data->scratch_tensor_index + 12;
  TfLiteTensor* grad_scale_intermediate;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/12,
                                              &grad_scale_intermediate));
  TfLiteIntArray* grad_scale_intermediate_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    grad_scale_intermediate_shape->data[i] = operand->dims->data[i];
  }

  grad_scale_intermediate->type = operand->type;
  grad_scale_intermediate->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, grad_scale_intermediate,
                                          grad_scale_intermediate_shape));

//Quantization part
   if (operand->type == kTfLiteInt8 || operand->type == kTfLiteInt16) {


     TfLiteIntArray* operand_dequantize_shape =
        TfLiteIntArrayCreate(operand->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
      operand_dequantize_shape->data[i] = operand->dims->data[i];
    }
    node->temporaries->data[13] = data->scratch_tensor_index+13;
    TfLiteTensor* operand_dequantize;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/13,
                                                &operand_dequantize));
    operand_dequantize->type = kTfLiteFloat32;
    operand_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, operand_dequantize,
                                                     operand_dequantize_shape));

         TfLiteIntArray* scale_dequantize_shape =
        TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < scale->dims->size; ++i) {
      scale_dequantize_shape->data[i] = scale->dims->data[i];
    }
    node->temporaries->data[14] = data->scratch_tensor_index+14;
    TfLiteTensor* scale_dequantize;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/14,
                                                &scale_dequantize));
    scale_dequantize->type = kTfLiteFloat32;
    scale_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scale_dequantize,
                                                     scale_dequantize_shape));


         TfLiteIntArray* mean_dequantize_shape =
        TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < scale->dims->size; ++i) {
      mean_dequantize_shape->data[i] = scale->dims->data[i];
    }
    node->temporaries->data[15] = data->scratch_tensor_index+15;
    TfLiteTensor* mean_dequantize;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/15,
                                                &mean_dequantize));
    mean_dequantize->type = kTfLiteFloat32;
    mean_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, mean_dequantize,
                                                     mean_dequantize_shape));

         TfLiteIntArray* variance_dequantize_shape =
        TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < scale->dims->size; ++i) {
      variance_dequantize_shape->data[i] = scale->dims->data[i];
    }
    node->temporaries->data[16] = data->scratch_tensor_index+16;
    TfLiteTensor* variance_dequantize;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/16,
                                                &variance_dequantize));
    variance_dequantize->type = kTfLiteFloat32;
    variance_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, variance_dequantize,
                                                     variance_dequantize_shape));

             TfLiteIntArray* grad_output_dequantize_shape =
        TfLiteIntArrayCreate(operand->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
       grad_output_dequantize_shape->data[i] = operand->dims->data[i];
    }
    node->temporaries->data[17] = data->scratch_tensor_index+17;
    TfLiteTensor* grad_output_dequantize;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/17,
                                                &grad_output_dequantize));
    grad_output_dequantize->type = kTfLiteFloat32;
    grad_output_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, grad_output_dequantize,
                                                     grad_output_dequantize_shape));

             TfLiteIntArray* grad_operand_dequantize_shape =
        TfLiteIntArrayCreate(operand->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
       grad_operand_dequantize_shape->data[i] = operand->dims->data[i];
    }
    node->temporaries->data[18] = data->scratch_tensor_index+18;
    TfLiteTensor* grad_operand_dequantize;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/18,
                                                &grad_operand_dequantize));
    grad_operand_dequantize->type = kTfLiteFloat32;
    grad_operand_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, grad_operand_dequantize,
                                                     grad_operand_dequantize_shape));

                 TfLiteIntArray* grad_scale_dequantize_shape =
        TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < scale->dims->size; ++i) {
       grad_scale_dequantize_shape->data[i] = scale->dims->data[i];
    }
    node->temporaries->data[19] = data->scratch_tensor_index+19;
    TfLiteTensor* grad_scale_dequantize;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/19,
                                                &grad_scale_dequantize));
    grad_scale_dequantize->type = kTfLiteFloat32;
    grad_scale_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, grad_scale_dequantize,
                                                     grad_scale_dequantize_shape));

                   TfLiteIntArray* grad_offset_dequantize_shape =
        TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < scale->dims->size; ++i) {
       grad_offset_dequantize_shape->data[i] = scale->dims->data[i];
    }
    node->temporaries->data[20] = data->scratch_tensor_index+20;
    TfLiteTensor* grad_offset_dequantize;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/20,
                                                &grad_offset_dequantize));
    grad_offset_dequantize->type = kTfLiteFloat32;
    grad_offset_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, grad_offset_dequantize,
                                                     grad_offset_dequantize_shape));

    
   }                                        
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 5);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 3);

  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, OpData::kOperandTensor, &operand));

  const TfLiteTensor* scale;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, OpData::kScaleTensor, &scale));

  const TfLiteTensor* mean;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, OpData::kMeanTensor, &mean));

  const TfLiteTensor* variance;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, OpData::kVarianceTensor, &variance));

  const TfLiteTensor* grad_output;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, OpData::kGradOutputTensor, &grad_output));

  TfLiteTensor* grad_operand;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, OpData::kOutputGradOperandTensor,
                             &grad_operand));

  TfLiteTensor* grad_scale;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, OpData::kOutputGradScaleTensor,
                                  &grad_scale));

  TfLiteTensor* grad_offset;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, OpData::kOutputGradOffsetTensor,
                             &grad_offset));

  const TfLiteBatchNormGradParams* params =
      reinterpret_cast<TfLiteBatchNormGradParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // (C1) 0 <= feature_index < rank(operand).
  int operand_rank = NumDimensions(operand);
  TF_LITE_ENSURE(context, params->feature_index >= 0 &&
                              params->feature_index < operand_rank);

  // (C2) Ensure all tensors have the same baseline element type.
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, scale->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, mean->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, variance->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, grad_output->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, grad_operand->type);
  TF_LITE_ENSURE_TYPES_EQ(context, scale->type, grad_scale->type);
  TF_LITE_ENSURE_TYPES_EQ(context, scale->type, grad_offset->type);

  // (C3) Ensure operand, grad_output, and grad_operand have the same shape.
  TF_LITE_ENSURE_EQ(
      context, TfLiteIntArrayEqual(operand->dims, grad_output->dims), true);

  // (C5) size(scale) = dim(operand, feature_index)
  TF_LITE_ENSURE_EQ(context, scale->dims->data[0],
                    operand->dims->data[params->feature_index]);

  TfLiteIntArray* grad_operand_size = TfLiteIntArrayCopy(operand->dims);
  TfLiteIntArray* grad_scale_size = TfLiteIntArrayCreate(1);
  grad_scale_size->data[0] = operand->dims->data[params->feature_index];
  TfLiteIntArray* grad_offset_size = TfLiteIntArrayCreate(1);
  grad_offset_size->data[0] = operand->dims->data[params->feature_index];
  // constraint check as per stablehlo spec - pending

  // Prepare temporary tensors
  TF_LITE_ENSURE_OK(
      context, PrepareTemporaries(context, node, params, operand, grad_output,scale));
  // Resize output tensors
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, grad_operand, grad_operand_size));
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, grad_scale, grad_scale_size));
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, grad_offset, grad_offset_size));

  TF_LITE_ENSURE_EQ(
      context, TfLiteIntArrayEqual(operand->dims, grad_operand->dims), true);

  // (C4) Ensure scale, mean, variance, grad_scale, and grad_offset have the
  // same shape.
  TF_LITE_ENSURE_EQ(context, TfLiteIntArrayEqual(scale->dims, mean->dims),
                    true);
  TF_LITE_ENSURE_EQ(context, TfLiteIntArrayEqual(scale->dims, variance->dims),
                    true);
  TF_LITE_ENSURE_EQ(context, TfLiteIntArrayEqual(scale->dims, grad_scale->dims),
                    true);
  TF_LITE_ENSURE_EQ(context,
                    TfLiteIntArrayEqual(scale->dims, grad_offset->dims), true);

  return kTfLiteOk;
}

template <typename DataType>
void BroadcastInDim(TfLiteContext* context, const TfLiteTensor* input,
                    const TfLiteIntArray* dims, TfLiteTensor* output) {
  // Update output type and dimensions
  output->type = input->type;

  // Perform broadcasting manually if necessary
  const DataType* input_data = GetTensorData<DataType>(input);
  DataType* output_data = GetTensorData<DataType>(output);
  int size = NumElements(output);
  std::cout << "size " << size << std::endl;
  int input_size = NumElements(input);
  std::cout << "input size broadcast in dim " << input_size << std::endl;

  for (int i = 0; i < size; ++i) {
    output_data[i] = input_data[i % input_size];
  }
}

}  // namespace



template <typename DataType>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                      const TfLiteTensor* operand, const TfLiteTensor* scale,
                      const TfLiteTensor* mean, const TfLiteTensor* variance,
                      const TfLiteTensor* grad_output,
                      TfLiteTensor* grad_operand, TfLiteTensor* grad_scale,
                      TfLiteTensor* grad_offset) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* epsilon_tensor = GetTemporary(context, node, 0);
  TfLiteTensor* centered_operand = GetTemporary(context, node, 1);
  TfLiteTensor* stddev = GetTemporary(context, node, 2);
  TfLiteTensor* normalized_operand = GetTemporary(context, node, 3);
  TfLiteTensor* elements_per_feature_tensor = GetTemporary(context, node, 4);
  TfLiteTensor* i1 = GetTemporary(context, node, 5);
  TfLiteTensor* i4 = GetTemporary(context, node, 6);
  TfLiteTensor* i5 = GetTemporary(context, node, 7);
  TfLiteTensor* i6 = GetTemporary(context, node, 8);
  TfLiteTensor* grad_output_centered_operand_mul =
      GetTemporary(context, node, 9);
  TfLiteTensor* grad_output_reduced = GetTemporary(context, node, 10);
  TfLiteTensor* i3_intermediate = GetTemporary(context, node, 11);
  TfLiteTensor* grad_scale_intermediate = GetTemporary(context, node, 12);

  // Set the value of epsilon in the tensor data
  const TfLiteBatchNormGradParams* params =
      reinterpret_cast<TfLiteBatchNormGradParams*>(node->builtin_data);
  // Set feature index and epsilon
  const int feature_index = params->feature_index;
  const float epsilon = params->epsilon;

  TfLiteIntArray* feature_dims = TfLiteIntArrayCreate(1);
  feature_dims->data[0] = feature_index;

  // Broadcast scale, mean, variance, and epsilon to match operand

  // Assuming scale is a TfLiteTensor pointer
  const DataType* scale_data = GetTensorData<DataType>(scale);
  int scale_size = NumElements(scale);

  epsilon_tensor->data.f[0] = epsilon;

  // Compute intermediate values
  ArithmeticParams op_params;
  op_params.broadcast_category = BroadcastableOpCategory::kGenericBroadcast;

  DataType* centered_operand_buffer = GetTensorData<DataType>(centered_operand);
  const float* operand_buffer = GetTensorData<float>(operand);
  const DataType* mean_data = GetTensorData<DataType>(mean);
  for (int i = 0; i < NumElements(centered_operand); ++i) {
    centered_operand_buffer[i] = static_cast<DataType>(
        operand_buffer[i] - mean_data[i % NumElements(mean)]);
  }

  int num_elements = NumElements(stddev);

  const DataType* variance_data = GetTensorData<DataType>(variance);
  int variance_size = NumElements(variance);
  DataType* stddev_buffer = GetTensorData<DataType>(stddev);
  for (int i = 0; i < NumElements(stddev); ++i) {
    stddev_buffer[i] = static_cast<DataType>(
        std::sqrt(variance_data[i % (NumElements(variance))] +
                  static_cast<DataType>(epsilon)));
    std::cout << "stddev  " << stddev_buffer[i] << std::endl;
  }
  for (int i = 0; i < NumElements(centered_operand); ++i) {
    std::cout << "centered operand  " << centered_operand_buffer[i]
              << std::endl;
  }
  float* normalized_buffer = GetTensorData<float>(normalized_operand);
  for (int i = 0; i < NumElements(normalized_operand); ++i) {
    normalized_buffer[i] = centered_operand_buffer[i] / stddev_buffer[i];
    std::cout << "normalized operand  " << normalized_buffer[i] << std::endl;
  }

  // Get the dimensions of the operand
  int operand_size = NumElements(operand);
  int feature_size = GetTensorShape(operand).Dims(feature_index);

  // Compute elements per feature
  float elements_per_feature = static_cast<float>(operand_size) / feature_size;
  elements_per_feature_tensor->data.f[0] = elements_per_feature;

  // // Broadcast to match the operand shape
  TfLiteIntArray* a = TfLiteIntArrayCreate(0);
  std::cout << "before broadcasting element per feature tensor  " << std::endl;
  // BroadcastInDim<DataType>(context, elements_per_feature_tensor, a,
  //                elements_per_feature_tensor_bcast);
  std::cout << "after broadcasting element per feature tensor  " << std::endl;
  //  DataType* element_per_feature_buffer =
  //       GetTensorData<DataType>(elements_per_feature_tensor_bcast);
  DataType* element_per_feature_tensor_buffer =
      GetTensorData<DataType>(elements_per_feature_tensor);
  const DataType* grad_output_buffer = GetTensorData<DataType>(grad_output);
  DataType* i1_buffer = GetTensorData<DataType>(i1);
  for (int i = 0; i < NumElements(i1); ++i) {
    i1_buffer[i] = grad_output_buffer[i] *
                   element_per_feature_tensor_buffer
                       [i % (NumElements(elements_per_feature_tensor))];
  }

  // //i2
  ComputeSum<DataType>(context, node, grad_output, feature_index,
                       grad_output_reduced);

  DataType* grad_output_centered_operand_mul_buffer =
      GetTensorData<DataType>(grad_output_centered_operand_mul);
  for (int i = 0; i < NumElements(grad_output_centered_operand_mul); ++i) {
    grad_output_centered_operand_mul_buffer[i] =
        grad_output_buffer[i] * centered_operand_buffer[i];
  }

  ComputeSum<DataType>(context, node, grad_output_centered_operand_mul,
                       feature_index, i3_intermediate);


  DataType* i4_buffer = GetTensorData<DataType>(i4);
  DataType* i3_intermediate_buffer = GetTensorData<DataType>(i3_intermediate);
  for (int i = 0; i < NumElements(i4); ++i) {
    i4_buffer[i] = i3_intermediate_buffer[i % (NumElements(i3_intermediate))] *
                   centered_operand_buffer[i];
    std::cout << "i4 buffer " << i4_buffer[i] << std::endl;
    std::cout << "i3 buffer " << i3_intermediate_buffer[i] << std::endl;
  }

  DataType* i5_buffer = GetTensorData<DataType>(i5);

  for (int i = 0; i < NumElements(i5); ++i) {
    i5_buffer[i] = i4_buffer[i] / (variance_data[i % (NumElements(variance))] +
                                   static_cast<DataType>(epsilon));
  }

  DataType* i6_buffer = GetTensorData<DataType>(i6);
  DataType* grad_output_reduced_buffer =
      GetTensorData<DataType>(grad_output_reduced);
  // std::cout << "num ele " << NumElements(i1) << "  " << NumElements(i2) << "
  // "
  //           << NumElements(i5) << "  " << NumElements(i6) << std::endl;
  for (int i = 0; i < NumElements(i6); ++i) {
    i6_buffer[i] =
        ((i1_buffer[i] -
          grad_output_reduced_buffer[i % NumElements(grad_output_reduced)]) -
         i5_buffer[i]);
    std::cout << "intermediate val " << i1_buffer[i] << " "
              << grad_output_reduced_buffer[i] << " " << i5_buffer[i]
              << std::endl;
    std::cout << "int 2 " << i1_buffer[i] - grad_output_reduced_buffer[i]
              << "   "
              << i1_buffer[i] - grad_output_reduced_buffer[i] - i5_buffer[i]
              << std::endl;
    std::cout << "i6 after cal " << i6_buffer[i] << std::endl;
  }

  for (int i = 0; i < NumElements(elements_per_feature_tensor); ++i) {
    std::cout << "element_per_feature data "
              << element_per_feature_tensor_buffer[i] << std::endl;
  }
  for (int i = 0; i < NumElements(i6); ++i) {
    std::cout << "i5 data " << i5_buffer[i] << std::endl;
    std::cout << "i1 data " << i1_buffer[i] << std::endl;
    // std::cout << "i2 data " << i2_buffer[i] << std::endl;
  }

  DataType* grad_operand_buffer = GetTensorData<DataType>(grad_operand);
  // DataType* scale_bcast_buffer = GetTensorData<DataType>(scale_bcast);
  for (int i = 0; i < NumElements(grad_operand); ++i) {
    grad_operand_buffer[i] =
        ((scale_data[i % scale_size] / stddev_buffer[i]) /
         element_per_feature_tensor_buffer
             [i % (NumElements(elements_per_feature_tensor))]) *
        i6_buffer[i];
  }

  DataType* grad_scale_intermediate_buffer =
      GetTensorData<DataType>(grad_scale_intermediate);
  for (int i = 0; i < NumElements(grad_scale_intermediate); ++i) {
    grad_scale_intermediate_buffer[i] =
        static_cast<DataType>(grad_output_buffer[i] * normalized_buffer[i]);
  }

  ComputeSum<DataType>(context, node, grad_scale_intermediate, feature_index,
                       grad_scale);

  // grad offset
  ComputeSum<DataType>(context, node, grad_output, feature_index, grad_offset);

  DataType* grad_offset_buffer = GetTensorData<DataType>(grad_offset);
  for (int i = 0; i < NumElements(grad_offset); ++i) {
    std::cout << "grad_offset data " << grad_offset_buffer[i] << std::endl;
  }
  TfLiteIntArrayFree(feature_dims);
  TfLiteIntArrayFree(a);
  return kTfLiteOk;
}

template <typename DataType>
TfLiteStatus EvalQuantImp(TfLiteContext* context, TfLiteNode* node,
                          const TfLiteTensor* operand,
                          const TfLiteTensor* scale, 
                          const TfLiteTensor* mean,
                          const TfLiteTensor* variance,
                          const TfLiteTensor* grad_output,
                          TfLiteTensor* grad_operand, TfLiteTensor* grad_scale,
                          TfLiteTensor* grad_offset) {
                            std::cout<<"entered eval quantiz"<<std::endl;

  TfLiteTensor* operand_dequantize = GetTemporary(context, node, 13);
  TfLiteTensor* scale_dequantize = GetTemporary(context, node, 14);
  TfLiteTensor* mean_dequantize = GetTemporary(context, node, 15);
  TfLiteTensor* variance_dequantize = GetTemporary(context, node, 16);
  TfLiteTensor* grad_output_dequantize = GetTemporary(context, node, 17);
  TfLiteTensor* grad_operand_dequantize = GetTemporary(context, node, 18);
  TfLiteTensor* grad_scale_dequantize = GetTemporary(context, node, 19);
  TfLiteTensor* grad_offset_dequantize = GetTemporary(context, node, 20);
 std::cout<<"entered eval quantiz fter 740"<<std::endl;
  dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
      context, node, operand, operand_dequantize);
       std::cout<<"entered eval quantiz after 743"<<std::endl;
  dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
      context, node, scale, scale_dequantize);
       std::cout<<"entered eval quantiz after 746"<<std::endl;
  dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
      context, node, mean, mean_dequantize);
       std::cout<<"entered eval quantiz after 749"<<std::endl;
  dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
      context, node, variance, variance_dequantize);
       std::cout<<"entered eval quantiz after 752"<<std::endl;
  dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
      context, node, grad_output, grad_output_dequantize);
 std::cout<<"entered eval quantiz before evalquantimp"<<std::endl;
  EvalImpl<float>(context,node,operand_dequantize, scale_dequantize, mean_dequantize, variance_dequantize, grad_output_dequantize, grad_operand_dequantize,
                  grad_scale_dequantize, grad_offset_dequantize);
                   std::cout<<"entered eval quantiz after evalquantimp"<<std::endl;

  RuntimeShape grad_operand_shape(GetTensorShape(grad_operand));
  RuntimeShape grad_operand_dequantize_shape(GetTensorShape(grad_operand_dequantize));
    RuntimeShape grad_scale_shape(GetTensorShape(grad_scale));
  RuntimeShape grad_scale_dequantize_shape(GetTensorShape(grad_scale_dequantize));
    RuntimeShape grad_offset_shape(GetTensorShape(grad_offset));
  RuntimeShape grad_offset_dequantize_shape(GetTensorShape(grad_offset_dequantize));
  std::cout<<"after shaPE 766"<<std::endl;
 
    tflite::QuantizationParams op_params;
    op_params.zero_point = grad_operand->params.zero_point;
    op_params.scale = grad_operand->params.scale;
    op_params.zero_point = grad_scale->params.zero_point;
    op_params.scale = grad_scale->params.scale;
    op_params.zero_point = grad_offset->params.zero_point;
    op_params.scale = grad_offset->params.scale;
    optimized_ops::AffineQuantize<DataType>(
        op_params, grad_operand_dequantize_shape,
        GetTensorData<float>(grad_operand_dequantize), grad_operand_shape,
        GetTensorData<DataType>(grad_operand));
        std::cout<<"AFTER 780"<<std::endl;
            optimized_ops::AffineQuantize<DataType>(
        op_params, grad_scale_dequantize_shape,
        GetTensorData<float>(grad_scale_dequantize), grad_scale_shape,
        GetTensorData<DataType>(grad_scale));
        std::cout<<"AFTER 784"<<std::endl;
            optimized_ops::AffineQuantize<DataType>(
        op_params, grad_offset_dequantize_shape,
        GetTensorData<float>(grad_offset_dequantize), grad_offset_shape,
        GetTensorData<DataType>(grad_offset));
  std::cout<<"end of evalqimp"<<std::endl;
   return kTfLiteOk;

}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  // Get the input tensors
  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, OpData::kOperandTensor, &operand));

  const TfLiteTensor* scale;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, OpData::kScaleTensor, &scale));

  const TfLiteTensor* mean;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, OpData::kMeanTensor, &mean));

  const TfLiteTensor* variance;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, OpData::kVarianceTensor, &variance));

  const TfLiteTensor* grad_output;
  TF_LITE_ENSURE_OK(
      context,
      GetInputSafe(context, node, OpData::kGradOutputTensor, &grad_output));

  // Get the output tensors
  TfLiteTensor* grad_operand;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, OpData::kOutputGradOperandTensor,
                             &grad_operand));

  TfLiteTensor* grad_scale;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, OpData::kOutputGradScaleTensor,
                                  &grad_scale));
  TfLiteTensor* grad_offset;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, OpData::kOutputGradOffsetTensor,
                             &grad_offset));

  switch (operand->type) {
    case kTfLiteFloat32: {
      return EvalImpl<float>(context, node, operand, scale, mean, variance,
                             grad_output, grad_operand, grad_scale,
                             grad_offset);
    }
    case kTfLiteFloat16: {
      return EvalImpl<Eigen::half>(context, node, operand, scale, mean,
                                   variance, grad_output, grad_operand,
                                   grad_scale, grad_offset);
    }
    case kTfLiteBFloat16: {
      return EvalImpl<Eigen::bfloat16>(context, node, operand, scale, mean,
                                       variance, grad_output, grad_operand,
                                       grad_scale, grad_offset);
    }
    case kTfLiteInt8: {
      std::cout<<"entered int 8"<<std::endl;
      return EvalQuantImp<int8_t>(context, node, operand, scale, mean, variance,
                                  grad_output, grad_operand, grad_scale,
                                  grad_offset);

    }
    case kTfLiteInt16: {
      return EvalQuantImp<int16_t>(context, node, operand, scale, mean,
                                   variance, grad_output, grad_operand,
                                   grad_scale, grad_offset);
    }
    default: {
      TF_LITE_KERNEL_LOG(
          context, "Type '%s' is not supported by stablehlo.batch_norm_grad.",
          TfLiteTypeGetName(operand->type));
      return kTfLiteError;
    }
  }
}
}  // namespace stablehlo_batch_norm_grad

// Registration function
TfLiteRegistration* Register_STABLEHLO_BATCH_NORM_GRAD() {
  static TfLiteRegistration r = {
      stablehlo_batch_norm_grad::Init, stablehlo_batch_norm_grad::Free,
      stablehlo_batch_norm_grad::Prepare, stablehlo_batch_norm_grad::Eval};
  std::cout << "inside reg" << std::endl;
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite