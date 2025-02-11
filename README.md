# Fire Detection with Faster R-CNN

This project uses a pre-trained Faster R-CNN model to detect fire in images. The model was trained to detect fire and will predict if a given image contains fire, visualizing the prediction with bounding boxes.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Example Images](#example-images)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)



## Installation
To get started clone the repository and install the required dependencies

## Usage

You can use the model to detect fire by running the 'DetectFire.detect' method in the 'inference.py' script

First instanciate the 'DetectFire()' class as something like: 'detector'

Then run the method as such: detector.detect(image_path)

The 'detect' method has other parameters such as 'graph_data' and 'threshold'

## Examples

example1.py demonstrates using the 'detect' method to get the highest score

example2.py demonstrates using the 'detect' method to get all the scores and bounding boxes above the threshold

example3.py demonstrates using the 'detect' methods' output together with opencv for a live inference

## Example Images

### Example 1: Candle Fire Detection
![Candle Fire Detection](examples/Candle_Predictions.png)

### Example 2: Forest Fire Detection
![Forest Fire Detection](examples/Forest_Predictions.png)

### Example 3: Kitchen Fire Detection
![Kitchen Fire Detection](examples/Kitchen_Predictions.png)

## Requirements

torch==1.13.1: PyTorch, for Torch tools

torchvision==0.14.1: TorchVision, for computer vision tools

matplotlib==3.7.0: Matplotlib, for visualization

Pillow==9.4.0: Pillow, for image manipulation

opencv-python==4.5.5.64: OpenCV, for live inference

numpy==1.21.6: NumPy, for manipulating tensors to better be used by other libraries

## License

This project is licensed under the CC0 1.0 Universal License

Please don't outright steal it :)