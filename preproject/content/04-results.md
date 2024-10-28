# Results

## Machine learning techniques for ISA detection

## CNN applied to binary code

Using CNN for analyzing binary machine code is not a novel idea. In this section, we will review the applications, datasets, and methods that have been previously explored within this domain.

### Encoding binary data

Traditionally, CNN is most widely used in visual tasks such as image classification, object detection, image segmentation, and computer vision [@cnn-survey]. Binary files, on the other hand, are just sequences of bytes, and exhibit no inherent multi-dimensionality in the same way as images or 3D objects do. Thus, a natural question to raise is: how do we convert a stream of bytes into a format that can be consumed by a CNN?

#### One-dimensional encoding

Rahul et al. [@Rahul2017] proposed a method for converting binary data into a one-dimensional input vector. They preprocess each file's binary content into a fixed-length vector of 128 decimal values. However, the paper does not elaborate on how this preprocessing is conducted. 

Li et al. [@Li2021] also takes a 1D encoding approach by converting each byte into a decimal value between 0 and 255. Then, the bytes are arranged as a one-dimensional vector and treated as a pixel value. The pixel vectors are finally compressed to a fixed size to ensure the input length is the same across instances. They experiment with vector lengths of 500, 1000, and 2000, and their evaluation shows that a longer input vector increases accuracy, while also increasing the computation time needed to train the model. They pick a vector length of 1000 as a trade-off between accuracy and computational efficiency.

Chaganti et al. [@Chaganti2022] uses the ELF header to locate the entry point of each binary program. From the entry point, 2000 bytes are extracted. If there are less than 2000 bytes present after the entry point, the remaining bytes are padded with zero values. They then run the bytes through an encryption cipher, perform base64 encoding of the encrypted bytes, and then run it through a word embedding layer before reaching the CNN.

TODO:

- 2D grayscale image
- 2D RGB image
- Recurrence plots
- Space-filling curves

### Transfer learning

TODO:

- VGG-16
- GoogLeNet
- ResNet-50

### CNN variations

TODO:

- XGBoost
- MFF-CNN
- S-DCNN
- Spatial CNN
- Bi-GRU-CNN
- Dual CNN

### Applications

#### Malware classification

TODO: Insert summary table

#### Compiler detection
