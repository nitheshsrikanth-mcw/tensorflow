#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/reference/broadcast_to.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/add.h"
#include "tensorflow/lite/kernels/internal/reference/sub.h"
#include "tensorflow/lite/kernels/internal/reference/mul.h"
#include "tensorflow/lite/kernels/internal/reference/div.h"
#include<iostream>

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_batch_norm_grad {
 constexpr int kMaxTemporaryTensors=15;
struct OpData {
  int scratch_tensor_index;
  // int feature_index;
  // float epsilon;
};

// Initialize function for the op
void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    std::cout<<"inside bef op init"<<std::endl;
  OpData* data = new OpData;
        std::cout<<"inside init "<<std::endl;
  
  // // Initialize feature dimensions
  // feature_dims = TfLiteIntArrayCreate(1);
  return data;
}

// Free function for the op
void Free(TfLiteContext* context, void* buffer) {
  OpData* data = reinterpret_cast<OpData*>(buffer);

  // Free allocated memory for feature dimensions
  // if (data->feature_dims != nullptr) {
  //   TfLiteIntArrayFree(data->feature_dims);
  // }
}

template <typename T>
void Sqrt(const T* input_data, T* output_data, int num_elements) {
  for (int i = 0; i < num_elements; ++i) {
    output_data[i] = std::sqrt(input_data[i]);
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {

  std::cout<<"inside pre"<<std::endl;

  /*operand dims creation*/
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
   const TfLiteTensor* operand = GetInput(context, node, 0);
    std::cout<<"bef op dims"<<std::endl;
  const TfLiteIntArray* operand_dims = operand->dims;
    context->AddTensors(context, kMaxTemporaryTensors,
                      &data->scratch_tensor_index);
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(15);
    node->temporaries->data[0] = data->scratch_tensor_index;


/*scale bcast creation*/
     std::cout<<"bef scale bcast"<<std::endl;
    TfLiteIntArray* scale_bcast_shape =
        TfLiteIntArrayCreate(operand->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
      std::cout<<"data of operand data i-"<<operand_dims->data[i]<<std::endl;
      scale_bcast_shape->data[i] = operand->dims->data[i];
      std::cout<<"data of scale  data i-"<<scale_bcast_shape->data[i]<<std::endl;
    }
    node->temporaries->data[0] = data->scratch_tensor_index;
    TfLiteTensor* scale_bcast;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/0,
                                                &scale_bcast));
    scale_bcast->type = kTfLiteFloat32;
    scale_bcast->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scale_bcast,
                                                     scale_bcast_shape));


/*mean bcast creation*/
std::cout<<"bef mean bcast"<<std::endl;
    TfLiteIntArray* mean_bcast_shape =
        TfLiteIntArrayCreate(operand->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
      mean_bcast_shape->data[i] = operand->dims->data[i];
    }
    node->temporaries->data[1] = data->scratch_tensor_index + 1;
    TfLiteTensor* mean_bcast;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/1,
                                                &mean_bcast));
    mean_bcast->type = kTfLiteFloat32;
    mean_bcast->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, mean_bcast,
                                            mean_bcast_shape));


/*variance bcast creation*/
std::cout<<"bef variance bcast"<<std::endl;
    TfLiteIntArray* variance_bcast_shape =
        TfLiteIntArrayCreate(operand->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
      variance_bcast_shape->data[i] = operand->dims->data[i];
    }
    node->temporaries->data[2] = data->scratch_tensor_index + 2;
    TfLiteTensor* variance_bcast;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/2,
                                                &variance_bcast));
    variance_bcast->type = kTfLiteFloat32;
    variance_bcast->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, variance_bcast,
                                                     variance_bcast_shape));


/*epsilon bacst */
  std::cout<<"bef epsilon bcast"<<std::endl;
      TfLiteIntArray* epsilon_bcast_shape =
        TfLiteIntArrayCreate(operand->dims->size);
    for (int i = 0; i < operand->dims->size; ++i) {
      std::cout<<"enter for loop "<<i<<std::endl;
      epsilon_bcast_shape->data[i] = operand->dims->data[i];
    }
    std::cout<<"after for loop"<<std::endl;
    node->temporaries->data[3] = data->scratch_tensor_index + 3;
    TfLiteTensor* epsilon_bcast;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/3,
                                                &epsilon_bcast));
    epsilon_bcast->type = kTfLiteFloat32;
    epsilon_bcast->allocation_type = kTfLiteArenaRw;
    std::cout<<"bef resize"<<std::endl;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, epsilon_bcast,
                                                     epsilon_bcast_shape));
                                                     std::cout<<"after resize"<<std::endl;
   
      for (int i = 0; i < operand->dims->size; ++i) {
      std::cout<<" epsilon "<<epsilon_bcast_shape->data[i]<<"variance  "<<variance_bcast_shape->data[i]<<std::endl;
   
    }
    std::cout<<"variance bcast dims size "<<variance_bcast->dims->size<<" epsilon bcast dims size "<<epsilon_bcast->dims->size<<std::endl; 


    std::cout << "bef epsilon tensor setup" << std::endl;

// Create shape array for epsilon_bcast matching the operand's dimensions
TfLiteIntArray* epsilon_tensor_shape = TfLiteIntArrayCreate(operand->dims->size);
for (int i = 0; i < operand->dims->size; ++i) {
    std::cout << "enter for loop " << i << std::endl;
    epsilon_tensor_shape->data[i] = operand->dims->data[i];
}
std::cout << "after for loop" << std::endl;
// Set up epsilon_tensor
node->temporaries->data[4] = data->scratch_tensor_index + 4;  // Adjust temporary index if needed
TfLiteTensor* epsilon_tensor;
TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/4, &epsilon_tensor));
epsilon_tensor->type = kTfLiteFloat32;  // Match this to operand->type if necessary
epsilon_tensor->allocation_type = kTfLiteArenaRw;

std::cout << "bef resize epsilon tensor" << std::endl;
TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, epsilon_tensor, epsilon_tensor_shape));
std::cout << "after resize epsilon tensor" << std::endl;
std::cout << "after resize epsilon tensor num ele" << NumElements(epsilon_tensor)<<std::endl;


// // Assuming the type is float, update this part based on the actual data type if it's different
// for (int i = 0; i < epsilon_tensor->dims->data[0]; ++i) {
//     epsilon_tensor->data.f[i] = epsilon;  // Assuming a flat layout for simplicity
// }

// Log the shape and contents of the tensor for debugging
for (int i = 0; i < epsilon_tensor->dims->size; ++i) {
    std::cout << "epsilon_tensor shape[" << i << "]: " << epsilon_tensor->dims->data[i] << std::endl;
}

std::cout << "after epsilon tensor setup" << std::endl;



/* centered operand bcast*/
     std::cout << "bef centered_operand bcast" << std::endl;
TfLiteIntArray* centered_operand_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
for (int i = 0; i < operand->dims->size; ++i) {
  std::cout << "enter centered_operand loop " << i << std::endl;
  centered_operand_bcast_shape->data[i] = operand->dims->data[i];
}
std::cout << "after centered_operand loop" << std::endl;

node->temporaries->data[5] = data->scratch_tensor_index + 5;  // Assuming the next available index is 4
TfLiteTensor* centered_operand;
TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/5, &centered_operand));
centered_operand->type = kTfLiteFloat32;
centered_operand->allocation_type = kTfLiteArenaRw;

std::cout << "bef resize centered_operand" << std::endl;
TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, centered_operand, centered_operand_bcast_shape));
std::cout << "after resize centered_operand" << std::endl; 


/*stddev bcast*/
std::cout << "bef stddev bcast" << std::endl;
TfLiteIntArray* stddev_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
for (int i = 0; i < operand->dims->size; ++i) {
  std::cout << "enter stddev loop " << i << std::endl;
  stddev_bcast_shape->data[i] = operand->dims->data[i];
}
std::cout << "after stddev loop" << std::endl;
node->temporaries->data[6] = data->scratch_tensor_index + 6;  // Assuming the next available index is 5
TfLiteTensor* stddev;
TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/6, &stddev));
stddev->type = kTfLiteFloat32;
stddev->allocation_type = kTfLiteArenaRw;
std::cout << "bef resize stddev" << std::endl;
TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, stddev, stddev_bcast_shape));
std::cout << "after resize stddev" << std::endl;

// 1. Allocate and resize tensor for normalized_operand
std::cout << "bef normalized_operand bcast" << std::endl;
TfLiteIntArray* normalized_operand_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
for (int i = 0; i < operand->dims->size; ++i) {
  std::cout << "enter normalized_operand loop " << i << std::endl;
  normalized_operand_bcast_shape->data[i] = operand->dims->data[i];
}
std::cout << "after normalized_operand loop" << std::endl;
node->temporaries->data[7] = data->scratch_tensor_index + 7;
TfLiteTensor* normalized_operand;
TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/7, &normalized_operand));
normalized_operand->type = kTfLiteFloat32;
normalized_operand->allocation_type = kTfLiteArenaRw;
std::cout << "bef resize normalized_operand" << std::endl;
TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, normalized_operand, normalized_operand_bcast_shape));
std::cout << "after resize normalized_operand" << std::endl;

// 2. Allocate and resize tensor for elements_per_feature_tensor
std::cout << "bef elements_per_feature_tensor bcast" << std::endl;
TfLiteIntArray* elements_per_feature_tensor_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
for (int i = 0; i < operand->dims->size; ++i) {
  std::cout << "enter elements_per_feature_tensor loop " << i << std::endl;
  elements_per_feature_tensor_bcast_shape->data[i] = operand->dims->data[i];
}
std::cout << "after elements_per_feature_tensor loop" << std::endl;
node->temporaries->data[8] = data->scratch_tensor_index + 8;
TfLiteTensor* elements_per_feature_tensor;
TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/8, &elements_per_feature_tensor));
elements_per_feature_tensor->type = kTfLiteFloat32;
elements_per_feature_tensor->allocation_type = kTfLiteArenaRw;
std::cout << "bef resize elements_per_feature_tensor" << std::endl;
TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, elements_per_feature_tensor, elements_per_feature_tensor_bcast_shape));
std::cout << "after resize elements_per_feature_tensor" << std::endl;

// 3. Allocate and resize tensor for grad_operand
std::cout << "bef grad_operand bcast" << std::endl;
TfLiteIntArray* grad_operand_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
for (int i = 0; i < operand->dims->size; ++i) {
  std::cout << "enter grad_operand loop " << i << std::endl;
  grad_operand_bcast_shape->data[i] = operand->dims->data[i];
}
std::cout << "after grad_operand loop" << std::endl;
node->temporaries->data[9] = data->scratch_tensor_index + 9;
TfLiteTensor* grad_operand;
TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/9, &grad_operand));
grad_operand->type = kTfLiteFloat32;
grad_operand->allocation_type = kTfLiteArenaRw;
std::cout << "bef resize grad_operand" << std::endl;
TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, grad_operand, grad_operand_bcast_shape));
std::cout << "after resize grad_operand" << std::endl;

// 4. Allocate and resize tensor for i1
std::cout << "bef i1 bcast" << std::endl;
TfLiteIntArray* i1_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
for (int i = 0; i < operand->dims->size; ++i) {
  std::cout << "enter i1 loop " << i << std::endl;
  i1_bcast_shape->data[i] = operand->dims->data[i];
}
std::cout << "after i1 loop" << std::endl;
node->temporaries->data[10] = data->scratch_tensor_index + 10;
TfLiteTensor* i1;
TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/10, &i1));
i1->type = kTfLiteFloat32;
i1->allocation_type = kTfLiteArenaRw;
std::cout << "bef resize i1" << std::endl;
TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, i1, i1_bcast_shape));
std::cout << "after resize i1" << std::endl;

// 5. Allocate and resize tensor for grad_output_centered_operand_mu
std::cout << "bef grad_output_centered_operand_mu bcast" << std::endl;
TfLiteIntArray* grad_output_centered_operand_mu_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
for (int i = 0; i < operand->dims->size; ++i) {
  std::cout << "enter grad_output_centered_operand_mu loop " << i << std::endl;
  grad_output_centered_operand_mu_bcast_shape->data[i] = operand->dims->data[i];
}
std::cout << "after grad_output_centered_operand_mu loop" << std::endl;
node->temporaries->data[11] = data->scratch_tensor_index + 11;
TfLiteTensor* grad_output_centered_operand_mu;
TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/11, &grad_output_centered_operand_mu));
grad_output_centered_operand_mu->type = kTfLiteFloat32;
grad_output_centered_operand_mu->allocation_type = kTfLiteArenaRw;
std::cout << "bef resize grad_output_centered_operand_mu" << std::endl;
TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, grad_output_centered_operand_mu, grad_output_centered_operand_mu_bcast_shape));
std::cout << "after resize grad_output_centered_operand_mu" << std::endl;

// 6. Allocate and resize tensor for grad_operand_centered_operand
std::cout << "bef grad_operand_centered_operand bcast" << std::endl;
TfLiteIntArray* grad_operand_centered_operand_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
for (int i = 0; i < operand->dims->size; ++i) {
  std::cout << "enter grad_operand_centered_operand loop " << i << std::endl;
  grad_operand_centered_operand_bcast_shape->data[i] = operand->dims->data[i];
}
std::cout << "after grad_operand_centered_operand loop" << std::endl;
node->temporaries->data[12] = data->scratch_tensor_index + 12;
TfLiteTensor* grad_operand_centered_operand;
TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/12, &grad_operand_centered_operand));
grad_operand_centered_operand->type = kTfLiteFloat32;
grad_operand_centered_operand->allocation_type = kTfLiteArenaRw;
std::cout << "bef resize grad_operand_centered_operand" << std::endl;
TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, grad_operand_centered_operand, grad_operand_centered_operand_bcast_shape));
std::cout << "after resize grad_operand_centered_operand" << std::endl;

// 7. Allocate and resize tensor for grad_scale
std::cout << "bef grad_scale bcast" << std::endl;
TfLiteIntArray* grad_scale_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
for (int i = 0; i < operand->dims->size; ++i) {
  std::cout << "enter grad_scale loop " << i << std::endl;
  grad_scale_bcast_shape->data[i] = operand->dims->data[i];
}
std::cout << "after grad_scale loop" << std::endl;
node->temporaries->data[13] = data->scratch_tensor_index + 13;
TfLiteTensor* grad_scale;
TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/13, &grad_scale));
grad_scale->type = kTfLiteFloat32;
grad_scale->allocation_type = kTfLiteArenaRw;
std::cout << "bef resize grad_scale" << std::endl;
TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, grad_scale, grad_scale_bcast_shape));
std::cout << "after resize grad_scale" << std::endl;

// 8. Allocate and resize tensor for grad_offset
std::cout << "bef grad_offset bcast" << std::endl;
TfLiteIntArray* grad_offset_bcast_shape = TfLiteIntArrayCreate(operand->dims->size);
for (int i = 0; i < operand->dims->size; ++i) {
  std::cout << "enter grad_offset loop " << i << std::endl;
  grad_offset_bcast_shape->data[i] = operand->dims->data[i];
}
std::cout << "after grad_offset loop" << std::endl;
node->temporaries->data[14] = data->scratch_tensor_index + 14;
TfLiteTensor* grad_offset;
TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/14, &grad_offset));
grad_offset->type = kTfLiteFloat32;
grad_offset->allocation_type = kTfLiteArenaRw;
std::cout << "bef resize grad_offset" << std::endl;
TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, grad_offset, grad_offset_bcast_shape));
std::cout << "after resize grad_offset" << std::endl;


  return kTfLiteOk;
}

// Manual sum reduction
void ComputeSum(TfLiteContext* context, const TfLiteTensor* input, int feature_index, TfLiteTensor* output) {
  TfLiteIntArray* reduction_dims = TfLiteIntArrayCreate(1);
  reduction_dims->data[0] = feature_index;
  
  // Create a tensor for output
  output->type = input->type;
  output->dims = TfLiteIntArrayCopy(input->dims);
  
  // Zero out the output tensor
  memset(output->data.raw, 0, NumElements(output) * sizeof(float));

  // Perform reduction manually
  const float* input_data = GetTensorData<float>(input);
  float* output_data = GetTensorData<float>(output);
  int size = 1;
  for (int i = 0; i < input->dims->size; ++i) {
    if (i != feature_index) {
      size *= input->dims->data[i];
    }
  }
  for (int i = 0; i < size; ++i) {
    output_data[i] += input_data[i];
  }
}

// Manual broadcasting
// void BroadcastInDim(TfLiteContext* context, const TfLiteTensor* input, const TfLiteIntArray* dims, TfLiteTensor* output) {
//   output->type = input->type;
//   output->dims = TfLiteIntArrayCopy(dims);

//   // Perform broadcasting manually
//   const float* input_data = GetTensorData<float>(input);
//   float* output_data = GetTensorData<float>(output);
//   int size = NumElements(output);
//   int input_size = NumElements(input);
//   for (int i = 0; i < size; ++i) {
//     output_data[i] = input_data[i % input_size];
//   }
// }
void BroadcastInDim(TfLiteContext* context, const TfLiteTensor* input, const TfLiteIntArray* dims, TfLiteTensor* output) {
    // Update output type and dimensions
    output->type = input->type;
    
    // Print the input tensor dimensions and size
    std::cout << "Input tensor dimensions: ";
    for (int i = 0; i < input->dims->size; ++i) {
        std::cout << input->dims->data[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Input tensor size: " << NumElements(input) << std::endl;
    
//     //Resize the output tensor to the target broadcast dimensions
//     TfLiteStatus status = context->ResizeTensor(context, output, TfLiteIntArrayCopy(dims));
//     if (status != kTfLiteOk) {
//         context->ReportError(context, "Failed to resize tensor for broadcasting.");
//         return;
//     }
// std::cout<<"dims size "<<dims->size<<std::endl;
//     // Print the output tensor dimensions and size after resizing
//     std::cout << "Output tensor dimensions after resize: ";
//     for (int i = 0; i < output->dims->size; ++i) {
//         std::cout << output->dims->data[i] << " ";
//     }

    std::cout << "Output tensor size after resize: " << NumElements(output) << std::endl;

    // Perform broadcasting manually if necessary
    const float* input_data = GetTensorData<float>(input);
    float* output_data = GetTensorData<float>(output);
    int size = NumElements(output);
    int input_size = NumElements(input);
    
    // Print the sizes used for broadcasting
    std::cout << "Performing broadcasting with size: " << size << " and input size: " << input_size << std::endl;
    
    for (int i = 0; i < size; ++i) {
        output_data[i] = input_data[i % input_size];
    }
}



TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  std::cout<<"inside eval"<<std::endl;
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
   

  // Get the input tensors
  const TfLiteTensor* operand = GetInput(context, node, 0);
  const TfLiteTensor* scale = GetInput(context, node, 1);
  std::cout<<"size of scale after creating temp tensor"<<scale->dims->size<<std::endl;
  const TfLiteTensor* mean = GetInput(context, node, 2);
  const TfLiteTensor* variance = GetInput(context, node, 3);
  const TfLiteTensor* grad_output = GetInput(context, node, 4);

  // Get the output tensors
  TfLiteTensor* output_grad_operand = GetOutput(context, node, 0);
  TfLiteTensor* output_grad_scale = GetOutput(context, node, 1);
  TfLiteTensor* output_grad_offset = GetOutput(context, node, 2);

  //Temporary tensor
  TfLiteTensor* scale_bcast = GetTemporary(context, node, 0);
  std::cout<<"size of scale_bcast after creating temp tensor"<<scale_bcast->dims->size<<std::endl;
  TfLiteTensor* mean_bcast = GetTemporary(context, node, 1);
  TfLiteTensor* variance_bcast = GetTemporary(context, node, 2);
  TfLiteTensor* epsilon_bcast = GetTemporary(context, node, 3);
  std::cout<<"after temp tensor"<<std::endl;

// Set the value of epsilon in the tensor data
const TfLiteBatchNormGradParams* params =
      reinterpret_cast<TfLiteBatchNormGradParams*>(
          node->builtin_data);
  // Set feature index and epsilon
  const int feature_index = params->feature_index;
  const float epsilon = params->epsilon;

   TfLiteIntArray* feature_dims=TfLiteIntArrayCreate(1);
  feature_dims->data[0]=feature_index;
  std::cout<<"feature dims size"<<feature_dims->size<<std::endl;

  // Broadcast scale, mean, variance, and epsilon to match operand
 

  TfLiteTensor* epsilon_tensor = GetTemporary(context, node, 4);
    if (epsilon_tensor == nullptr) {
    std::cout << "epsilon tensor is null!" << std::endl;
    return kTfLiteError;
}
else{
  std::cout<<"epsilon tensor is not null"<<std::endl;
}

  
std::cout<<"bef broadcast"<<std::endl;
  std::cout<<"Beforebcast scale size "<<scale_bcast->dims->size<<std::endl;
    std::cout<<"before bcast mean size "<<mean_bcast->dims->size<<std::endl;
  std::cout<<"before bcast variance size "<<variance_bcast->dims->size<<std::endl;
    std::cout<<"before bcast epsilon size "<<epsilon_bcast->dims->size<<std::endl;
    // Assuming scale is a TfLiteTensor pointer
const float* scale_data = GetTensorData<float>(scale_bcast);
int scale_size = NumElements(scale_bcast);

std::cout << "Values inside the scale_bcast tensor before bcast tensor: ";
for (int i = 0; i < scale_size; ++i) {
    std::cout << scale_data[i] << " ";
}
std::cout << std::endl;

  BroadcastInDim(context, scale, feature_dims, scale_bcast);
  const float* scale_data1 = GetTensorData<float>(scale_bcast);
int scale_size1 = NumElements(scale_bcast);

std::cout << "Values inside the scale_bcast tensor after bcast tensor: ";
for (int i = 0; i < scale_size1; ++i) {
    std::cout << scale_data1[i] << " ";
}
std::cout << std::endl;

  BroadcastInDim(context, mean, feature_dims, mean_bcast);
  BroadcastInDim(context, variance, feature_dims, variance_bcast);

  std::cout<<"epsilon tensor dims"<<(epsilon_tensor->dims->data==nullptr)<<std::endl;
  epsilon_tensor->dims->data[0] =epsilon;
  std::cout << "after resize epsilon tensor num ele in eva" << NumElements(epsilon_tensor)<<std::endl;
  BroadcastInDim(context, epsilon_tensor, feature_dims, epsilon_bcast);
 
  std::cout<<"after broadcast"<<std::endl;
  std::cout<<"Afterbcast scale size "<<scale_bcast->dims->size<<std::endl;
    std::cout<<"Afterbcast mean size "<<mean_bcast->dims->size<<std::endl;
  std::cout<<"Afterbcast variance size "<<variance_bcast->dims->size<<std::endl;
    std::cout<<"Afterbcast epsilon size "<<epsilon_bcast->dims->size<<std::endl;


  // Compute intermediate values
  ArithmeticParams op_params;
  op_params.broadcast_category = BroadcastableOpCategory::kGenericBroadcast;

  TfLiteTensor* centered_operand = GetTemporary(context, node, 5);
  if (centered_operand == nullptr) {
    std::cout << "Error: Tensor at index 5 (centered_operand) is null!" << std::endl;
    return kTfLiteError;
}
else{
  std::cout << "Error: Tensor at index 5 (centered_operand) is not null!" << std::endl;
}
  std::cout<<"bef sub"<<std::endl;
  reference_ops::Sub(op_params,
                     GetTensorShape(operand), GetTensorData<float>(operand),
                     GetTensorShape(mean_bcast), GetTensorData<float>(mean_bcast),
                     GetTensorShape(centered_operand), GetTensorData<float>(centered_operand));
                     std::cout<<"after sub"<<std::endl;

  std::cout << "Debug: Checking temporary tensor allocation at index 6" << std::endl;
TfLiteTensor* stddev = GetTemporary(context, node, 6);
std::cout<<"stddev size "<<stddev->dims->size<<std::endl;

if (stddev == nullptr) {
    std::cout << "Error: Temporary tensor at index 6 is not allocated!" << std::endl;
    return kTfLiteError;
}
else{
std::cout << "Debug: Temporary tensor at index 6 is allocated successfully" << std::endl;
}
std::cout<<"variance size - "<<variance_bcast->dims->size<<" epsilon bcast - "<<epsilon_bcast->dims->size<<std::endl;
std::cout<<"num ele variance size - "<<NumElements(variance_bcast)<<" epsilon bcast - "<<NumElements(epsilon_bcast)<<std::endl;

  reference_ops::Add(op_params,
                     GetTensorShape(variance_bcast), GetTensorData<float>(variance_bcast),
                     GetTensorShape(epsilon_bcast), GetTensorData<float>(epsilon_bcast),
                     GetTensorShape(stddev), GetTensorData<float>(stddev));
                      std::cout<<"after stddev size "<<stddev->dims->size<<std::endl;
   int num_elements = NumElements(stddev);
  Sqrt(GetTensorData<float>(stddev), GetTensorData<float>(stddev), num_elements);
  std::cout<<"after sqrt- stddev size - "<<stddev->dims->size<<std::endl;
  std::cout<<"after sqrt"<<std::endl;

  TfLiteTensor* normalized_operand = GetTemporary(context, node, 7);
  if (normalized_operand == nullptr) {
    std::cout << "Error: Temporary tensor at index 7 is not allocated!" << std::endl;
    return kTfLiteError;
}
else{
std::cout << "Debug: Temporary tensor at index 7 is allocated successfully" << std::endl;
}
  reference_ops::Div(op_params,
                     GetTensorShape(centered_operand), GetTensorData<float>(centered_operand),
                     GetTensorShape(stddev), GetTensorData<float>(stddev),
                     GetTensorShape(normalized_operand), GetTensorData<float>(normalized_operand));
                     std::cout<<"after div"<<std::endl;

  // Compute elements per feature
  std::cout<<"bef for loop in eval "<<std::endl;
  int elements_per_feature = 1;
  for (int i = 0; i < operand->dims->size; ++i) {
    std::cout<<"entering for loop"<<std::endl;
    if (i != feature_index) {
      std::cout<<"entering if"<<i<<std::endl;
      elements_per_feature *= operand->dims->data[i];
    }
  }
  std::cout<<"end for  loop"<<std::endl;

  TfLiteTensor* elements_per_feature_tensor = GetTemporary(context, node, 8);
  elements_per_feature_tensor->type = kTfLiteFloat32;
  elements_per_feature_tensor->dims = TfLiteIntArrayCreate(1);
  elements_per_feature_tensor->dims->data[0] = 1;
  elements_per_feature_tensor->data.f[0] = static_cast<float>(elements_per_feature);

  // Compute gradients
  TfLiteTensor* grad_operand = GetTemporary(context, node, 9);
  TfLiteTensor* i1 = GetTemporary(context, node, 10);
  ComputeSum(context, grad_output, feature_index, i1);
  BroadcastInDim(context, i1, feature_dims, i1);

  TfLiteTensor* grad_output_centered_operand_mul = GetTemporary(context, node, 11);
  reference_ops::Mul(op_params,
                     GetTensorShape(grad_output), GetTensorData<float>(grad_output),
                     GetTensorShape(centered_operand), GetTensorData<float>(centered_operand),
                     GetTensorShape(grad_output_centered_operand_mul), GetTensorData<float>(grad_output_centered_operand_mul));
  ComputeSum(context, grad_output_centered_operand_mul, feature_index, grad_output_centered_operand_mul);
  BroadcastInDim(context, grad_output_centered_operand_mul, feature_dims, grad_output_centered_operand_mul);

  TfLiteTensor* grad_operand_centered_operand = GetTemporary(context, node, 12);
  reference_ops::Div(op_params,
                     GetTensorShape(i1), GetTensorData<float>(i1),
                     GetTensorShape(stddev), GetTensorData<float>(stddev),
                     GetTensorShape(grad_operand_centered_operand), GetTensorData<float>(grad_operand_centered_operand));

  reference_ops::Mul(op_params,
                     GetTensorShape(grad_operand_centered_operand), GetTensorData<float>(grad_operand_centered_operand),
                     GetTensorShape(centered_operand), GetTensorData<float>(centered_operand),
                     GetTensorShape(grad_operand), GetTensorData<float>(grad_operand));

  ComputeSum(context, grad_operand, feature_index, grad_operand);

  TfLiteTensor* grad_scale = GetTemporary(context, node, 13);
  reference_ops::Mul(op_params,
                     GetTensorShape(grad_output), GetTensorData<float>(grad_output),
                     GetTensorShape(normalized_operand), GetTensorData<float>(normalized_operand),
                     GetTensorShape(grad_scale), GetTensorData<float>(grad_scale));
  ComputeSum(context, grad_scale, feature_index, grad_scale);

  TfLiteTensor* grad_offset = GetTemporary(context, node, 14);
  ComputeSum(context, grad_output, feature_index, grad_offset);

   // Copy results to outputs
  if (output_grad_operand->data.raw != grad_operand->data.raw) {
    memcpy(output_grad_operand->data.raw, grad_operand->data.raw, NumElements(output_grad_operand) * sizeof(float));
  }
  if (output_grad_scale->data.raw != grad_scale->data.raw) {
    memcpy(output_grad_scale->data.raw, grad_scale->data.raw, NumElements(output_grad_scale) * sizeof(float));
  }
  if (output_grad_offset->data.raw != grad_offset->data.raw) {
    memcpy(output_grad_offset->data.raw, grad_offset->data.raw, NumElements(output_grad_offset) * sizeof(float));
  }
std::cout<<"end of eval"<<std::endl;
  return kTfLiteOk;
}
}  // namespace stablehlo_batch_norm_grad

// Registration function
TfLiteRegistration* Register_STABLEHLO_BATCH_NORM_GRAD() {
  static TfLiteRegistration r = {
      stablehlo_batch_norm_grad::Init, stablehlo_batch_norm_grad::Free, stablehlo_batch_norm_grad::Prepare, stablehlo_batch_norm_grad::Eval
  };
    std::cout<<"inside reg"<<std::endl;
  return &r;
}


}  // namespace builtin
}  // namespace ops
}  // namespace tflite