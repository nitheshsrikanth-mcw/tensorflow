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
#include "gmock/gmock.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class BatchNormGradOpModel : public SingleOpModel {
 public:
  BatchNormGradOpModel(const TensorData& operand, const TensorData& scale,
                       const TensorData& mean, const TensorData& variance,
                       const TensorData& grad_output,
                       const TensorData& grad_operand,
                       const TensorData& grad_scale,
                       const TensorData& grad_offset,
                       TfLiteBatchNormGradParams params) {
    operand_ = AddInput(operand);
    scale_ = AddInput(scale);
    mean_ = AddInput(mean);
    variance_ = AddInput(variance);
    grad_output_ = AddInput(grad_output);

    grad_operand_ = AddOutput(grad_operand);
    grad_scale_ = AddOutput(grad_scale);
    grad_offset_ = AddOutput(grad_offset);

    SetBuiltinOp(BuiltinOperator_STABLEHLO_BATCH_NORM_GRAD,
                 BuiltinOptions2_StableHLOBatchNormGradOptions,
                 CreateStableHLOBatchNormGradOptions(builder_, params.epsilon,
                                                     params.feature_index)
                     .Union());

    BuildInterpreter({GetShape(operand_), GetShape(scale_), GetShape(mean_),
                      GetShape(variance_), GetShape(grad_output_)},
                     -1, false, false, false, false);

    AllocateAndDelegate(true);
  }

  int operand() { return operand_; }
  int scale() { return scale_; }
  int mean() { return mean_; }
  int variance() { return variance_; }
  int grad_output() { return grad_output_; }

  void SetInput(std::initializer_list<T> data) {
    PopulateTensor(operand_, data);
  }

  void SetScale(std::initializer_list<T> data) { PopulateTensor(scale_, data); }

  void SetMean(std::initializer_list<T> data) { PopulateTensor(mean_, data); }

  void SetVariance(std::initializer_list<T> data) {
    PopulateTensor(variance_, data);
  }

  void SetGradOutput(std::initializer_list<T> data) {
    PopulateTensor(grad_output_, data);
  }

  std::vector<T> GetOutputGradOperand() {
    return ExtractVector<T>(grad_operand_);
  }

  std::vector<T> GetOutputGradScale() { return ExtractVector<T>(grad_scale_); }

  std::vector<T> GetOutputGradOffset() {
    return ExtractVector<T>(grad_offset_);
  }

 protected:
  int operand_;
  int scale_;
  int mean_;
  int variance_;
  int grad_output_;
  int grad_operand_;
  int grad_scale_;
  int grad_offset_;
};

TEST(StableHLOBatchNormGradOpTest, BatchNormGradTestFloat32) {
  TfLiteBatchNormGradParams params = {0.0f, 2};

  BatchNormGradOpModel<float> model(
      {TensorType_FLOAT32, {2, 2, 2}}, {TensorType_FLOAT32, {2}},
      {TensorType_FLOAT32, {2}}, {TensorType_FLOAT32, {2}},
      {TensorType_FLOAT32, {2, 2, 2}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}}, params);

  model.SetInput({1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 4.0f, 1.0f, 2.0f});
  model.SetScale({1.0f, 1.0f});
  model.SetMean({2.0f, 3.0f});
  model.SetVariance({1.0f, 1.0f});
  model.SetGradOutput({0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutputGradOperand(),
      ElementsAreArray({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
  EXPECT_THAT(model.GetOutputGradScale(), ElementsAreArray({0.0f, 0.0f}));
  EXPECT_THAT(model.GetOutputGradOffset(), ElementsAreArray({0.4f, 0.4f}));
}

TEST(StableHLOBatchNormGradOpTest, BatchNormGradTestFloat32WithNonZeroEpsilon) {
  TfLiteBatchNormGradParams params = {1e-5, 2};

  BatchNormGradOpModel<float> model(
      {TensorType_FLOAT32, {2, 2, 3}}, {TensorType_FLOAT32, {3}},
      {TensorType_FLOAT32, {3}}, {TensorType_FLOAT32, {3}},
      {TensorType_FLOAT32, {2, 2, 3}}, {TensorType_FLOAT32, {}},
      {TensorType_FLOAT32, {}}, {TensorType_FLOAT32, {}}, params);

  model.SetInput({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                  11.0f, 12.0f});
  model.SetScale({1.0f, 1.0f, 1.0f});
  model.SetMean({2.0f, 3.0f, 4.0f});
  model.SetVariance({1.0f, 1.0f, 1.0f});
  model.SetGradOutput(
      {0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutputGradOperand(),
      ElementsAreArray(ArrayFloatNear(
          {0.6999895, 0.6999895, 0.6999895, -1.399979, -1.399979, -1.399979,
           -3.4999475, -3.4999475, -3.4999475, -5.599916, -5.599916, -5.599916},
          1e-3)));
  EXPECT_THAT(model.GetOutputGradScale(),
              ElementsAreArray(
                  ArrayFloatNear({2.799986f, 2.799986f, 2.799986f}, 1e-4)));
  EXPECT_THAT(model.GetOutputGradOffset(),
              ElementsAreArray({0.8f, 0.8f, 0.8f}));
}

TEST(StableHLOBatchNormGradOpTest, BatchNormGradTestFloat16) {
  {
    TfLiteBatchNormGradParams params = {0.01f, 2};

    BatchNormGradOpModel<Eigen::half> model(
        {TensorType_FLOAT16, {2, 2, 2}}, {TensorType_FLOAT16, {2}},
        {TensorType_FLOAT16, {2}}, {TensorType_FLOAT16, {2}},
        {TensorType_FLOAT16, {2, 2, 2}}, {TensorType_FLOAT16, {}},
        {TensorType_FLOAT16, {}}, {TensorType_FLOAT16, {}}, params);

    model.SetInput(
        {static_cast<Eigen::half>(1.0f), static_cast<Eigen::half>(2.0f),
         static_cast<Eigen::half>(3.0f), static_cast<Eigen::half>(4.0f),
         static_cast<Eigen::half>(3.0f), static_cast<Eigen::half>(4.0f),
         static_cast<Eigen::half>(5.0f), static_cast<Eigen::half>(6.0f)});
    model.SetScale(
        {static_cast<Eigen::half>(0.5f), static_cast<Eigen::half>(0.8f)});
    model.SetMean(
        {static_cast<Eigen::half>(1.5f), static_cast<Eigen::half>(2.5f)});
    model.SetVariance(
        {static_cast<Eigen::half>(0.3f), static_cast<Eigen::half>(0.7f)});
    model.SetGradOutput(
        {static_cast<Eigen::half>(0.3f), static_cast<Eigen::half>(0.3f),
         static_cast<Eigen::half>(0.3f), static_cast<Eigen::half>(0.3f),
         static_cast<Eigen::half>(0.3f), static_cast<Eigen::half>(0.3f),
         static_cast<Eigen::half>(0.3f), static_cast<Eigen::half>(0.3f)});

    model.Invoke();

    EXPECT_THAT(
        model.GetOutputGradOperand(),
        ElementsAreArray(ArrayFloatNear({static_cast<Eigen::half>(0.65179343),
                                         static_cast<Eigen::half>(0.30087422),
                                         static_cast<Eigen::half>(-1.9553803),
                                         static_cast<Eigen::half>(-0.90262267),
                                         static_cast<Eigen::half>(-1.9553803),
                                         static_cast<Eigen::half>(-0.90262267),
                                         static_cast<Eigen::half>(-4.56255404),
                                         static_cast<Eigen::half>(-2.10611956)},
                                        1e-5)));
    EXPECT_THAT(
        model.GetOutputGradScale(),
        ElementsAreArray(ArrayFloatNear({static_cast<Eigen::half>(3.23289544),
                                         static_cast<Eigen::half>(2.13620698)},
                                        1e-4)));
    EXPECT_THAT(
        model.GetOutputGradOffset(),
        ElementsAreArray(ArrayFloatNear(
            {static_cast<Eigen::half>(1.2f), static_cast<Eigen::half>(1.2f)})));
  }
}

TEST(StableHLOBatchNormGradOpTest, BatchNormGradTestBFloat16) {
  {
    TfLiteBatchNormGradParams params = {0.1f, 0};

    BatchNormGradOpModel<Eigen::bfloat16> model(
        {TensorType_BFLOAT16, {2, 2, 2}}, {TensorType_BFLOAT16, {2}},
        {TensorType_BFLOAT16, {2}}, {TensorType_BFLOAT16, {2}},
        {TensorType_BFLOAT16, {2, 2, 2}}, {TensorType_BFLOAT16, {}},
        {TensorType_BFLOAT16, {}}, {TensorType_BFLOAT16, {}}, params);

    model.SetInput(
        {static_cast<Eigen::bfloat16>(1.5f), static_cast<Eigen::bfloat16>(3.0f),
         static_cast<Eigen::bfloat16>(4.5f), static_cast<Eigen::bfloat16>(6.0f),
         static_cast<Eigen::bfloat16>(3.0f), static_cast<Eigen::bfloat16>(6.0f),
         static_cast<Eigen::bfloat16>(4.5f),
         static_cast<Eigen::bfloat16>(9.0f)});
    model.SetScale({static_cast<Eigen::bfloat16>(0.9f),
                    static_cast<Eigen::bfloat16>(1.2f)});
    model.SetMean({static_cast<Eigen::bfloat16>(2.0f),
                   static_cast<Eigen::bfloat16>(5.0f)});
    model.SetVariance({static_cast<Eigen::bfloat16>(0.4f),
                       static_cast<Eigen::bfloat16>(0.9f)});
    model.SetGradOutput(
        {static_cast<Eigen::bfloat16>(0.4f), static_cast<Eigen::bfloat16>(0.4f),
         static_cast<Eigen::bfloat16>(0.4f), static_cast<Eigen::bfloat16>(0.4f),
         static_cast<Eigen::bfloat16>(0.4f), static_cast<Eigen::bfloat16>(0.4f),
         static_cast<Eigen::bfloat16>(0.4f),
         static_cast<Eigen::bfloat16>(0.4f)});

    model.Invoke();

    EXPECT_THAT(model.GetOutputGradOperand(),
                ElementsAreArray(
                    ArrayFloatNear({static_cast<Eigen::bfloat16>(.12727922),
                                    static_cast<Eigen::bfloat16>(2.04),
                                    static_cast<Eigen::bfloat16>(-0.6363961),
                                    static_cast<Eigen::bfloat16>(-1.02),
                                    static_cast<Eigen::bfloat16>(-0.25455844),
                                    static_cast<Eigen::bfloat16>(-1.02),
                                    static_cast<Eigen::bfloat16>(-0.6363961),
                                    static_cast<Eigen::bfloat16>(-4.08)},
                                   1e-5)));
    EXPECT_THAT(model.GetOutputGradScale(),
                ElementsAreArray(
                    ArrayFloatNear({static_cast<Eigen::bfloat16>(0.73137085),
                                    static_cast<Eigen::bfloat16>(3.97989899)},
                                   1e-4)));
    EXPECT_THAT(model.GetOutputGradOffset(),
                ElementsAreArray({static_cast<Eigen::bfloat16>(1.6f),
                                  static_cast<Eigen::bfloat16>(1.6f)}));
  }
}

template <typename T>
class QuantizedBatchNormGradOpModel : public BatchNormGradOpModel<T> {
 public:
  QuantizedBatchNormGradOpModel(const TensorData& operand,
                                const TensorData& scale, const TensorData& mean,
                                const TensorData& variance,
                                const TensorData& grad_output,
                                const TensorData& grad_operand,
                                const TensorData& grad_scale,
                                const TensorData& grad_offset,
                                TfLiteBatchNormGradParams params)
      : BatchNormGradOpModel<T>(operand, scale, mean, variance, grad_output,
                                grad_operand, grad_scale, grad_offset, params) {
  }
  std::vector<float> GetDequantizedGradOperand() {
    return Dequantize<T>(this->template ExtractVector<T>(this->grad_operand_),
                         this->GetScale(this->grad_operand_),
                         this->GetZeroPoint(this->grad_operand_));
  }
  std::vector<float> GetDequantizedGradScale() {
    return Dequantize<T>(this->template ExtractVector<T>(this->grad_scale_),
                         this->GetScale(this->grad_scale_),
                         this->GetZeroPoint(this->grad_scale_));
  }
  std::vector<float> GetDequantizedGradOffset() {
    return Dequantize<T>(this->template ExtractVector<T>(this->grad_offset_),
                         this->GetScale(this->grad_offset_),
                         this->GetZeroPoint(this->grad_offset_));
  }
};

template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep = (max - min) / (std::numeric_limits<T>::max() -
                                        std::numeric_limits<T>::min());
  return 2.0 * kQuantizedStep;
}

TEST(StableHLOBatchNormGradOpTest, BatchNormGradQuantizedInt16) {
  {
    TfLiteBatchNormGradParams params = {0.0f, 2};
    float kQuantizedTolerance = GetTolerance<int16_t>(-127, 127);

    QuantizedBatchNormGradOpModel<int16_t> model(
        {TensorType_INT16, {2, 2, 2}, -127, 127},
        {TensorType_INT16, {2}, -127, 127}, {TensorType_INT16, {2}, -127, 127},
        {TensorType_INT16, {2}, -127, 127},
        {TensorType_INT16, {2, 2, 2}, -127, 127},
        {TensorType_INT16, {}, -127, 127}, {TensorType_INT16, {}, -127, 127},
        {TensorType_INT16, {}, -127, 127}, params);

    std::vector<float> operand = {1, 2, 3, 4, 3, 4, 1, 2};
    std::vector<float> scale = {1, 1};
    std::vector<float> mean = {1, 1};
    std::vector<float> variance = {1, 1};
    std::vector<float> gradoutput = {1, 1, 1, 1, 1, 1, 1, 1};
    model.QuantizeAndPopulate<int16_t>(model.operand(), operand);
    model.QuantizeAndPopulate<int16_t>(model.scale(), scale);
    model.QuantizeAndPopulate<int16_t>(model.mean(), mean);
    model.QuantizeAndPopulate<int16_t>(model.variance(), variance);
    model.QuantizeAndPopulate<int16_t>(model.grad_output(), gradoutput);

    model.Invoke();

    EXPECT_THAT(model.GetDequantizedGradOperand(),
                ElementsAreArray(ArrayFloatNear({0, -2, -2, -6, -2, -6, 0, -2.},
                                                kQuantizedTolerance)));
    EXPECT_THAT(model.GetDequantizedGradScale(),
                ElementsAreArray(ArrayFloatNear({4, 8}, kQuantizedTolerance)));
    EXPECT_THAT(model.GetDequantizedGradOffset(),
                ElementsAreArray(ArrayFloatNear({4, 4}, kQuantizedTolerance)));
  }
}

TEST(StableHLOBatchNormGradOpTest, BatchNormGradQuantizedInt8) {
  {
    TfLiteBatchNormGradParams params = {3.0f, 2};
    float kQuantizedTolerance = GetTolerance<int8_t>(-127, 127);

    QuantizedBatchNormGradOpModel<int8_t> model(
        {TensorType_INT8, {2, 2, 2}, -127, 127},
        {TensorType_INT8, {2}, -127, 127}, {TensorType_INT8, {2}, -127, 127},
        {TensorType_INT8, {2}, -127, 127},
        {TensorType_INT8, {2, 2, 2}, -127, 127},
        {TensorType_INT8, {}, -127, 127}, {TensorType_INT8, {}, -127, 127},
        {TensorType_INT8, {}, -127, 127}, params);

    std::vector<float> operand = {1, 2, 3, 4, 3, 4, 1, 2};
    std::vector<float> scale = {1, 1};
    std::vector<float> mean = {1, 1};
    std::vector<float> variance = {1, 1};
    std::vector<float> gradoutput = {1, 1, 1, 1, 1, 1, 1, 1};
    model.QuantizeAndPopulate<int8_t>(model.operand(), operand);
    model.QuantizeAndPopulate<int8_t>(model.scale(), scale);
    model.QuantizeAndPopulate<int8_t>(model.mean(), mean);
    model.QuantizeAndPopulate<int8_t>(model.variance(), variance);
    model.QuantizeAndPopulate<int8_t>(model.grad_output(), gradoutput);

    model.Invoke();

    EXPECT_THAT(model.GetDequantizedGradOperand(),
                ElementsAreArray(ArrayFloatNear(
                    {0, -0.25, -0.25, -0.75, -0.25, -0.75, 0, -0.25},
                    kQuantizedTolerance)));
    EXPECT_THAT(model.GetDequantizedGradScale(),
                ElementsAreArray(ArrayFloatNear({2, 4}, kQuantizedTolerance)));
    EXPECT_THAT(model.GetDequantizedGradOffset(),
                ElementsAreArray(ArrayFloatNear({4, 4}, kQuantizedTolerance)));
  }
}

}  // namespace
}  // namespace tflite