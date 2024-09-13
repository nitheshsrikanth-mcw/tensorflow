/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "Eigen/Core"
#include "tensorflow/lite/kernels/dequantize.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_batch_norm_grad {
namespace {

constexpr int kMaxTemporaryTensors = 18;
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
  enum { kGradOperandTensor, kGradScaleTensor, kGradOffsetTensor };
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
  for (int i = 0; i < operand_rank; ++i) {
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
  DataType* batch_sum_buffer = GetTensorData<DataType>(batch_sum);

  return kTfLiteOk;
}

TfLiteStatus PrepareTemporaries(TfLiteContext* context, TfLiteNode* node,
                                const TfLiteBatchNormGradParams* params,
                                const TfLiteTensor* operand,
                                const TfLiteTensor* grad_output,
                                const TfLiteTensor* scale) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  context->AddTensors(context, kMaxTemporaryTensors,
                      &data->scratch_tensor_index);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(kMaxTemporaryTensors);

  node->temporaries->data[0] = data->scratch_tensor_index;
  TfLiteTensor* epsilon_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 0, &epsilon_tensor));
  TfLiteIntArray* epsilon_tensor_shape = TfLiteIntArrayCreate(1);
  epsilon_tensor_shape->data[0] = 1;
  epsilon_tensor->type = operand->type;
  epsilon_tensor->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, epsilon_tensor,
                                                   epsilon_tensor_shape));

  node->temporaries->data[1] = data->scratch_tensor_index + 1;
  TfLiteTensor* centered_operand;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 1, &centered_operand));
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

  node->temporaries->data[2] = data->scratch_tensor_index + 2;
  TfLiteTensor* stddev;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, 2, &stddev));
  TfLiteIntArray* stddev_bcast_shape =
      TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    stddev_bcast_shape->data[i] = operand->dims->data[i];
  }
  stddev->type = operand->type;
  stddev->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, stddev, stddev_bcast_shape));

  node->temporaries->data[3] = data->scratch_tensor_index + 3;
  TfLiteTensor* normalized_operand;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 3, &normalized_operand));
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

  node->temporaries->data[4] = data->scratch_tensor_index + 4;
  TfLiteTensor* elements_per_feature_tensor;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, 4,
                                              &elements_per_feature_tensor));
  TfLiteIntArray* elements_per_feature_tensor_shape = TfLiteIntArrayCreate(0);

  elements_per_feature_tensor->type = operand->type;
  elements_per_feature_tensor->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, elements_per_feature_tensor,
                                          elements_per_feature_tensor_shape));

  node->temporaries->data[5] = data->scratch_tensor_index + 5;
  TfLiteTensor* i6;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, 5, &i6));
  TfLiteIntArray* i6_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
  for (int i = 0; i < operand->dims->size; ++i) {
    i6_bcast_shape->data[i] = operand->dims->data[i];
  }
  i6->type = operand->type;
  i6->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, i6, i6_bcast_shape));

  node->temporaries->data[6] = data->scratch_tensor_index + 6;
  TfLiteTensor* grad_output_centered_operand_mul;
  TF_LITE_ENSURE_OK(
      context,
      GetTemporarySafe(context, node, 6, &grad_output_centered_operand_mul));
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

  node->temporaries->data[7] = data->scratch_tensor_index + 7;
  TfLiteTensor* grad_output_reduced;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, 7, &grad_output_reduced));
  TfLiteIntArray* grad_output_reduced_shape = TfLiteIntArrayCreate(1);
  grad_output_reduced_shape->data[0] =
      grad_output->dims->data[params->feature_index];

  grad_output_reduced->type = operand->type;
  grad_output_reduced->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, grad_output_reduced,
                                                   grad_output_reduced_shape));

  node->temporaries->data[8] = data->scratch_tensor_index + 8;
  TfLiteTensor* i3;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, 8, &i3));
  TfLiteIntArray* i3_shape = TfLiteIntArrayCreate(1);
  i3_shape->data[0] =
      grad_output_centered_operand_mul->dims->data[params->feature_index];

  i3->type = operand->type;
  i3->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, i3, i3_shape));

  node->temporaries->data[9] = data->scratch_tensor_index + 9;
  TfLiteTensor* grad_scale_intermediate;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, 9, &grad_scale_intermediate));
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

  if (operand->type == kTfLiteInt8 || operand->type == kTfLiteInt16) {
    TfLiteIntArray* operand_dequantize_shape =
        TfLiteIntArrayCreate(operand->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
      operand_dequantize_shape->data[i] = operand->dims->data[i];
    }
    node->temporaries->data[10] = data->scratch_tensor_index + 10;
    TfLiteTensor* operand_dequantize;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, 10, &operand_dequantize));
    operand_dequantize->type = kTfLiteFloat32;
    operand_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, operand_dequantize,
                                            operand_dequantize_shape));

    TfLiteIntArray* scale_dequantize_shape =
        TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < scale->dims->size; ++i) {
      scale_dequantize_shape->data[i] = scale->dims->data[i];
    }
    node->temporaries->data[11] = data->scratch_tensor_index + 11;
    TfLiteTensor* scale_dequantize;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, 11, &scale_dequantize));
    scale_dequantize->type = kTfLiteFloat32;
    scale_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scale_dequantize,
                                                     scale_dequantize_shape));

    TfLiteIntArray* mean_dequantize_shape =
        TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < scale->dims->size; ++i) {
      mean_dequantize_shape->data[i] = scale->dims->data[i];
    }
    node->temporaries->data[12] = data->scratch_tensor_index + 12;
    TfLiteTensor* mean_dequantize;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, 12, &mean_dequantize));
    mean_dequantize->type = kTfLiteFloat32;
    mean_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, mean_dequantize,
                                                     mean_dequantize_shape));

    TfLiteIntArray* variance_dequantize_shape =
        TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < scale->dims->size; ++i) {
      variance_dequantize_shape->data[i] = scale->dims->data[i];
    }
    node->temporaries->data[13] = data->scratch_tensor_index + 13;
    TfLiteTensor* variance_dequantize;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, 13, &variance_dequantize));
    variance_dequantize->type = kTfLiteFloat32;
    variance_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, variance_dequantize,
                                            variance_dequantize_shape));

    TfLiteIntArray* grad_output_dequantize_shape =
        TfLiteIntArrayCreate(operand->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
      grad_output_dequantize_shape->data[i] = operand->dims->data[i];
    }
    node->temporaries->data[14] = data->scratch_tensor_index + 14;
    TfLiteTensor* grad_output_dequantize;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, 14, &grad_output_dequantize));
    grad_output_dequantize->type = kTfLiteFloat32;
    grad_output_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, grad_output_dequantize,
                                            grad_output_dequantize_shape));

    TfLiteIntArray* grad_operand_dequantize_shape =
        TfLiteIntArrayCreate(operand->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
      grad_operand_dequantize_shape->data[i] = operand->dims->data[i];
    }
    node->temporaries->data[15] = data->scratch_tensor_index + 15;
    TfLiteTensor* grad_operand_dequantize;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, 15, &grad_operand_dequantize));
    grad_operand_dequantize->type = kTfLiteFloat32;
    grad_operand_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, grad_operand_dequantize,
                                            grad_operand_dequantize_shape));

    TfLiteIntArray* grad_scale_dequantize_shape =
        TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < scale->dims->size; ++i) {
      grad_scale_dequantize_shape->data[i] = scale->dims->data[i];
    }
    node->temporaries->data[16] = data->scratch_tensor_index + 16;
    TfLiteTensor* grad_scale_dequantize;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, 16, &grad_scale_dequantize));
    grad_scale_dequantize->type = kTfLiteFloat32;
    grad_scale_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, grad_scale_dequantize,
                                            grad_scale_dequantize_shape));

    TfLiteIntArray* grad_offset_dequantize_shape =
        TfLiteIntArrayCreate(scale->dims->size);
    for (int i = 0; i < scale->dims->size; ++i) {
      grad_offset_dequantize_shape->data[i] = scale->dims->data[i];
    }
    node->temporaries->data[17] = data->scratch_tensor_index + 17;
    TfLiteTensor* grad_offset_dequantize;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, 17, &grad_offset_dequantize));
    grad_offset_dequantize->type = kTfLiteFloat32;
    grad_offset_dequantize->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, grad_offset_dequantize,
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
      context,
      GetOutputSafe(context, node, OpData::kGradOperandTensor, &grad_operand));

  TfLiteTensor* grad_scale;
  TF_LITE_ENSURE_OK(
      context,
      GetOutputSafe(context, node, OpData::kGradScaleTensor, &grad_scale));

  TfLiteTensor* grad_offset;
  TF_LITE_ENSURE_OK(
      context,
      GetOutputSafe(context, node, OpData::kGradOffsetTensor, &grad_offset));

  const TfLiteBatchNormGradParams* params =
      reinterpret_cast<TfLiteBatchNormGradParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  int operand_rank = NumDimensions(operand);
  TF_LITE_ENSURE(context, params->feature_index >= 0 &&
                              params->feature_index < operand_rank);

  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, scale->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, mean->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, variance->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, grad_output->type);
  TF_LITE_ENSURE_TYPES_EQ(context, operand->type, grad_operand->type);
  TF_LITE_ENSURE_TYPES_EQ(context, scale->type, grad_scale->type);
  TF_LITE_ENSURE_TYPES_EQ(context, scale->type, grad_offset->type);

  TF_LITE_ENSURE_EQ(
      context, TfLiteIntArrayEqual(operand->dims, grad_output->dims), true);

  TF_LITE_ENSURE_EQ(context, scale->dims->data[0],
                    operand->dims->data[params->feature_index]);

  TfLiteIntArray* grad_operand_size = TfLiteIntArrayCopy(operand->dims);
  TfLiteIntArray* grad_scale_size = TfLiteIntArrayCreate(1);
  grad_scale_size->data[0] = operand->dims->data[params->feature_index];
  TfLiteIntArray* grad_offset_size = TfLiteIntArrayCreate(1);
  grad_offset_size->data[0] = operand->dims->data[params->feature_index];

  TF_LITE_ENSURE_OK(context, PrepareTemporaries(context, node, params, operand,
                                                grad_output, scale));
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, grad_operand, grad_operand_size));
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, grad_scale, grad_scale_size));
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, grad_offset, grad_offset_size));

  TF_LITE_ENSURE_EQ(
      context, TfLiteIntArrayEqual(operand->dims, grad_operand->dims), true);

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
  TfLiteTensor* i6 = GetTemporary(context, node, 5);
  TfLiteTensor* grad_output_centered_operand_mul =
      GetTemporary(context, node, 6);
  TfLiteTensor* grad_output_reduced = GetTemporary(context, node, 7);
  TfLiteTensor* i3 = GetTemporary(context, node, 8);
  TfLiteTensor* grad_scale_intermediate = GetTemporary(context, node, 9);

  const TfLiteBatchNormGradParams* params =
      reinterpret_cast<TfLiteBatchNormGradParams*>(node->builtin_data);

  const int64_t feature_index = params->feature_index;
  const float epsilon = params->epsilon;

  TfLiteIntArray* feature_dims = TfLiteIntArrayCreate(1);
  feature_dims->data[0] = feature_index;

  const DataType* scale_data = GetTensorData<DataType>(scale);
  int scale_size = NumElements(scale);

  epsilon_tensor->data.f[0] = epsilon;

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
  }

  DataType* normalized_buffer = GetTensorData<DataType>(normalized_operand);
  for (int i = 0; i < NumElements(normalized_operand); ++i) {
    normalized_buffer[i] = centered_operand_buffer[i] / stddev_buffer[i];
  }

  int operand_size = NumElements(operand);
  int feature_size = GetTensorShape(operand).Dims(feature_index);

  DataType elements_per_feature =
      static_cast<DataType>(operand_size / feature_size);
  // elements_per_feature_tensor->data.f[0] = elements_per_feature;

  TfLiteIntArray* a = TfLiteIntArrayCreate(0);

  DataType* element_per_feature_tensor_buffer =
      GetTensorData<DataType>(elements_per_feature_tensor);
  element_per_feature_tensor_buffer[0] = elements_per_feature;
  const DataType* grad_output_buffer = GetTensorData<DataType>(grad_output);

  ComputeSum<DataType>(context, node, grad_output, feature_index,
                       grad_output_reduced);

  DataType* grad_output_centered_operand_mul_buffer =
      GetTensorData<DataType>(grad_output_centered_operand_mul);
  for (int i = 0; i < NumElements(grad_output_centered_operand_mul); ++i) {
    grad_output_centered_operand_mul_buffer[i] =
        grad_output_buffer[i] * centered_operand_buffer[i];
  }

  ComputeSum<DataType>(context, node, grad_output_centered_operand_mul,
                       feature_index, i3);

  DataType* i3_buffer = GetTensorData<DataType>(i3);

  DataType* i6_buffer = GetTensorData<DataType>(i6);
  DataType* grad_output_reduced_buffer =
      GetTensorData<DataType>(grad_output_reduced);

  for (int i = 0; i < NumElements(i6); ++i) {
    i6_buffer[i] =
        ((grad_output_buffer[i] *
              element_per_feature_tensor_buffer
                  [i % (NumElements(elements_per_feature_tensor))] -
          grad_output_reduced_buffer[i % NumElements(grad_output_reduced)]) -
         (i3_buffer[i % (NumElements(i3))] * centered_operand_buffer[i]) /
             (variance_data[i % (NumElements(variance))] +
              static_cast<DataType>(epsilon)));
  }

  DataType* grad_operand_buffer = GetTensorData<DataType>(grad_operand);
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

  ComputeSum<DataType>(context, node, grad_output, feature_index, grad_offset);

  DataType* grad_offset_buffer = GetTensorData<DataType>(grad_offset);

  TfLiteIntArrayFree(feature_dims);
  TfLiteIntArrayFree(a);
  return kTfLiteOk;
}

template <typename DataType>
TfLiteStatus EvalQuantImp(TfLiteContext* context, TfLiteNode* node,
                          const TfLiteTensor* operand,
                          const TfLiteTensor* scale, const TfLiteTensor* mean,
                          const TfLiteTensor* variance,
                          const TfLiteTensor* grad_output,
                          TfLiteTensor* grad_operand, TfLiteTensor* grad_scale,
                          TfLiteTensor* grad_offset) {
  TfLiteTensor* operand_dequantize = GetTemporary(context, node, 10);
  TfLiteTensor* scale_dequantize = GetTemporary(context, node, 11);
  TfLiteTensor* mean_dequantize = GetTemporary(context, node, 12);
  TfLiteTensor* variance_dequantize = GetTemporary(context, node, 13);
  TfLiteTensor* grad_output_dequantize = GetTemporary(context, node, 14);
  TfLiteTensor* grad_operand_dequantize = GetTemporary(context, node, 15);
  TfLiteTensor* grad_scale_dequantize = GetTemporary(context, node, 16);
  TfLiteTensor* grad_offset_dequantize = GetTemporary(context, node, 17);
  dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
      context, node, operand, operand_dequantize);
  dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
      context, node, scale, scale_dequantize);
  dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
      context, node, mean, mean_dequantize);
  dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
      context, node, variance, variance_dequantize);
  dequantize::DequantizeImpl<dequantize::KernelType::kGenericOptimized>(
      context, node, grad_output, grad_output_dequantize);
  EvalImpl<float>(context, node, operand_dequantize, scale_dequantize,
                  mean_dequantize, variance_dequantize, grad_output_dequantize,
                  grad_operand_dequantize, grad_scale_dequantize,
                  grad_offset_dequantize);

  RuntimeShape grad_operand_shape(GetTensorShape(grad_operand));
  RuntimeShape grad_operand_dequantize_shape(
      GetTensorShape(grad_operand_dequantize));
  RuntimeShape grad_scale_shape(GetTensorShape(grad_scale));
  RuntimeShape grad_scale_dequantize_shape(
      GetTensorShape(grad_scale_dequantize));
  RuntimeShape grad_offset_shape(GetTensorShape(grad_offset));
  RuntimeShape grad_offset_dequantize_shape(
      GetTensorShape(grad_offset_dequantize));

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
  optimized_ops::AffineQuantize<DataType>(
      op_params, grad_scale_dequantize_shape,
      GetTensorData<float>(grad_scale_dequantize), grad_scale_shape,
      GetTensorData<DataType>(grad_scale));

  optimized_ops::AffineQuantize<DataType>(
      op_params, grad_offset_dequantize_shape,
      GetTensorData<float>(grad_offset_dequantize), grad_offset_shape,
      GetTensorData<DataType>(grad_offset));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
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
      context,
      GetOutputSafe(context, node, OpData::kGradOperandTensor, &grad_operand));

  TfLiteTensor* grad_scale;
  TF_LITE_ENSURE_OK(
      context,
      GetOutputSafe(context, node, OpData::kGradScaleTensor, &grad_scale));
  TfLiteTensor* grad_offset;
  TF_LITE_ENSURE_OK(
      context,
      GetOutputSafe(context, node, OpData::kGradOffsetTensor, &grad_offset));

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

TfLiteRegistration* Register_STABLEHLO_BATCH_NORM_GRAD() {
  static TfLiteRegistration r = {
      stablehlo_batch_norm_grad::Init, stablehlo_batch_norm_grad::Free,
      stablehlo_batch_norm_grad::Prepare, stablehlo_batch_norm_grad::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite