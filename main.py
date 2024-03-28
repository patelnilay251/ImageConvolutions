import numpy as np

def convolute(image, kernel, depthwise=True):
  # Extract image dimensions
  input_height, input_width, num_channels = image.shape

  # Extract kernel dimensions (assuming square kernel)
  kernel_size = kernel.shape[0]

  # Pad the image for border handling (assuming zero padding)
  pad_size = int(kernel_size // 2)
  padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')

  # Initialize output image
  output_height = input_height - kernel_size + 1
  output_width = input_width - kernel_size + 1
  output = np.zeros((output_height, output_width, num_channels if depthwise else kernel.shape[-1]))

  # Call _standard_convolution with the padded image and kernel
  output = _standard_convolution(padded_image, kernel)

  return output

def _standard_convolution(image, kernel):
  # Extract image and kernel dimensions
  input_height, input_width = image.shape[:2]
  kernel_size = kernel.shape[0]

  # Allocate memory for the output before the loop
  output = np.zeros((input_height - kernel_size + 1, input_width - kernel_size + 1))

  # Iterate over output positions
  for y in range(input_height - kernel_size + 1):
    for x in range(input_width - kernel_size + 1):
      # Extract image patch for convolution
      image_patch = image[y:y+kernel_size, x:x+kernel_size]

      # Perform element-wise multiplication and summation
      output[y, x] = np.sum(image_patch * kernel)

  return output

# Example usage
image = np.random.rand(5, 5, 3)  # Sample image (5x5x3)
kernel = np.ones((3, 3))  # Sample kernel (3x3)

# Depthwise convolution
depthwise_output = convolute(image.copy(), kernel.copy())

# Pointwise convolution (assuming kernel depth matches input channels)
pointwise_output = convolute(image.copy(), kernel.copy(), depthwise=False)

print("Original image:", image)
print("Depthwise convolution output:", depthwise_output)
print("Pointwise convolution output:", pointwise_output)
