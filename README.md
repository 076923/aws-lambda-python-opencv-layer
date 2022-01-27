## Introduction

This repository provides `pre-built headless OpenCV` for AWS Lambda. Python OpenCV includes native code libraries, I compiled and build these libraries using a Linux development machine so that the binaries are compatible with Amazon Linux. Once this layer is applied, you can use the layer to host your own algorithms on lambda for Python OpenCV. Also, This layer file contains version `Numpy==1.21.5`. These libraries were created in `Python 3.7.10`.

## Apply Layer

1. Upload the `opencv-python-headless.zip` file to Amazon S3.

2. Open the Layers page of the Lambda console.

3. Choose Create layer.

4. Under Layer configuration, for Name, enter a name for your layer.

5. To upload OpenCV Python Headless layer

6. Choose Upload a file from Amazon S3. Then, for Amazon S3 link URL, enter a link to the file.

7. Choose Create.

## Support Pre-built Version

- opencv-python-headless-3.4.17.61
- opencv-python-headless-4.0.1.24
- opencv-python-headless-4.1.2.30
- opencv-python-headless-4.2.0.34
- opencv-python-headless-4.3.0.38
- opencv-python-headless-4.4.0.46
- opencv-python-headless-4.5.5.62

## Structure

Example file structure for the `opencv-python-headless` library:
```
opencv-python-headless.{major}.{minor}.{revision}.{package_version}.zip
  └ python/bin
  └ python/cv2
  └ python/numpy
  └ python/numpy.libs
  └ python/numpy-1.21.5.dist-info
  └ python/opencv_python_headless-{major}.{minor}.{revision}.{package_version}.dist-info
  └ python/opencv_python_headless.libs
```

## Another layer

The following [link][lambda-link] show how you can structure the folders in your layer .zip archive.

[lambda-link]: https://docs.aws.amazon.com/en_es/lambda/latest/dg/configuration-layers.html

## Authors

Daehee Yun(s076923@gmail.com)