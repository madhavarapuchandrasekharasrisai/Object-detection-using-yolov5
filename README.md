# 🚧 Object Detection with Darknet and TensorFlow 🚧

This project is a Python implementation of the [YOLOv3 object detection algorithm](https://pjreddie.com/darknet/yolo/) using [TensorFlow](https://www.tensorflow.org/) and the [Darknet neural network framework](https://pjreddie.com/darknet/). It is built on top of the [Darkflow](https://github.com/balancer-team/darkflow) project, which is a Python interface for Darknet.

![Screenshot](https://raw.githubusercontent.com/elliotwu/darkflow-tensorflow/master/screenshot.png)

## ✨ Features

* Object detection using the YOLOv3 algorithm
* Pre-trained weights for the [COCO dataset](http://cocodataset.org/#home)
* Customizable classes and anchors
* Real-time object detection
* Scalable image size
* Multi-threading for faster processing
* Customizable colors for detected objects

## 💡 Enhancements

* Improved preprocessing and postprocessing functions
* Added functions for generating colors and scaling boxes
* Added support for custom classes and anchors
* Improved error handling and logging

## 🛠 Technologies

* [TensorFlow](https://www.tensorflow.org/)
* [Darknet](https://pjreddie.com/darknet/)
* [OpenCV](https://opencv.org/)
* [Pillow](https://pillow.readthedocs.io/en/stable/)
* [NumPy](https://numpy.org/)

## 🚀 Getting Started

1. Clone this repository:

```bash
git clone https://github.com/elliotwu/darkflow-tensorflow.git
cd darkflow-tensorflow
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Download the [YOLOv3 weights](https://pjreddie.com/media/files/yolov3.weights) and place it in the `weights` directory.

4. Download the [COCO dataset classes](https://github.com/pjreddie/darknet/blob/master/data/coco.names) and place it in the `data` directory.

5. Download the [anchors](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) and place it in the `data` directory.

6. Run the object detection script:

```bash
python detect.py --image <image_path> --model yolov3 --weights yolov3.weights --classes coco.names --anchors yolov3.cfg
```

## 🧑‍💻 Usage

The `detect.py` script can be used to detect objects in an image. The following command line arguments are available:

* `--image`: The path to the image to be processed.
* `--model`: The model to be used for object detection. Currently supports `yolov3`.
* `--weights`: The path to the weights file for the model.
* `--classes`: The path to the classes file.
* `--anchors`: The path to the anchors file.
* `--threshold`: The confidence threshold for object detection.
* `--gpu`: The GPU ID to be used for processing.
* `--model_image_size`: The size of the input image for the model.

## 🔄 Data Handling

The `preprocess_image` function scales and normalizes the input image to be compatible with the YOLOv3 model. The `draw_boxes` function draws the detected objects on the input image.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgments

* [AlexeyAB](https://github.com/AlexeyAB) for the [Darkflow](https://github.com/balancer-team/darkflow) project.
* [Joseph Redmon](https://pjreddie.com/) for the [YOLOv3 algorithm](https://pjreddie.com/darknet/yolo/).
* [Pavel Yakubovskiy](https://github.com/pavelyakubovskiy) for the [Real-time Object Detection with TensorFlow and YOLOv3](https://github.com/pavelyakubovskiy/yolo_tensorflow) project.
* [OpenCV](https://opencv.org/) for the image processing functions.
* [Pillow](https://pillow.readthedocs.io/en/stable/) for the image manipulation functions.
* [NumPy](https://numpy.org/) for the array manipulation functions.
