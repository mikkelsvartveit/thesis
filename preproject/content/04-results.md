# Results

## Machine learning techniques for ISA detection

## CNN applied to binary code

Using CNN for analyzing binary machine code is not a novel idea. In this section, we will review the applications, datasets, and methods that have been previously explored within this domain.

### Encoding binary data

Traditionally, CNN is widely used in visual tasks such as image classification, object detection, image segmentation, and computer vision [@cnn-survey]. Binary files, on the other hand, are just sequences of bytes, and exhibit no inherent multi-dimensionality in the same way as images or 3D objects do. Thus, a natural question to raise is: how do we convert a stream of bytes into a format that can be consumed by a CNN?

#### One-dimensional vector

Rahul et al. [@Rahul2017] proposed a method for converting binary data into a one-dimensional input vector. They preprocessed each file's binary content into a fixed-length vector of 128 decimal values. However, the paper does not elaborate on how this preprocessing is conducted.

Li et al. [@Li2021] also took a 1D encoding approach by converting each byte into a decimal value between 0 and 255. The bytes were then arranged as a one-dimensional vector and treated as a pixel value. Finally, the pixel vectors were compressed to a fixed size to ensure the input length stays the same across instances. They experimented with vector lengths of 500, 1000, and 2000, and their evaluation showed that while a longer input length increased accuracy, it also increased the computation time needed to train the model. They picked a vector length of 1000 as a trade-off between accuracy and computational efficiency.

Chaganti et al. [@Chaganti2022] used the ELF header to locate the entry point of each binary program. From this entry point, 2000 bytes were extracted. If there were less than 2000 bytes present after the entry point, the remaining bytes were padded with zero values. They then ran the bytes through an encryption cipher, performed base64 encoding of the encrypted bytes, and then used a word embedding layer before reaching the CNN.

#### Two-dimensional grayscale image

The most common way to convert a binary file into a format interpretable by a CNN is to encode it as an image.

Kumari et al. [@Kumari2017] extracted each byte from the file and represented them as unsigned integers. Then, based on the file size, the integers were arranged in a 2D array with an aspect ratio close to 1:1. Each value was treated as a grayscale pixel with values between 0 and 255. Finally, the image was resized to 150x150 pixels to ensure identical input dimensions across instances. Prima & Bouhorma [@Prima2021], Hammad et al. [@Hammad2022], and Al-Masri et al. [@Al-Masri2024] used similar square image encodings, albeit with slightly different image sizes.

A variation of the square image is to use images of a fixed width, but variable length. Yang et al. [@Yang2018] fixed the image width to 512 pixels and let the height depend on the file size. El-Shafai et al. [@El-Shafai2021] took this a step further by defining a table of different width values based on the file size. Files over 1000 KB would get a width of 1024 pixels, files between 500 and 1000 KB would get a width of 768 pixels, and so on, with smaller files getting proportionally smaller widths. The latter approach was also used by Alvee et al. [@Alvee2021], Liang et al. [@Liang2021], and Son et al. [@Son2022].

TODO:

- 2D RGB image
- Recurrence plots
- Space-filling curves

### Transfer learning

Transfer learning is a machine learning technique where a model developed for one task is re-used for another task. Transfer learning is very useful for cases where there is not a lot of training data available, as well as in cases of limited computation power or time. Using a transfer learning approach can allow for deep networks despite these constraints.

Kumari et al. [@Kumari2017] experimented with two different transfer learning approaches in addition to training their own, significantly shallower, model from scratch. In the first approach, they used a pre-trained VGG-16 model and removed its fully connected layers at the end. Re-using only the convolutional blocks, they added and trained a small fully-connected network on top of the network. For the second approach, they also fine-tuned the last convolutional block before adding the same fully-connected network on top. Their evaluations showed that the pre-trained VGG-16 model with no fine-tuning performed the best among their three approaches, proving that using pre-trained CNN models optimized for images also can perform well for other applications. Prima et al. [@Prima2021] also did transfer learning by using VGG-16 with a fully connected block at the end. They found that the transfer learning model achieved the same performance as their from-scratch CNN, but a limitation of the paper is that they do not outline the architecture of the model they trained from scratch.

El-Shafai et al. [@El-Shafai2021] compared the performance of eight different pre-trained CNN models. They used transfer learning with fine-tuning, and found the best performing model to be VGG-16. It achieved a striking 99.97% accuracy for malware classification on the MalImg dataset, while reducing the number of training parameters by 99.92% compared to training VGG-16 from scratch.

Hammad et al. [@Hammad2022] used a pre-trained GoogLeNet model, which is designed with computational efficiency and memory footprint in mind. Compared to VGG-16's 138 million, GoogLeNet only needs 4 million parameters. To make predictions, the authors used a basic k-nearest neighbors (KNN) classifier on top. A notable feature of KNN models is that they do not requiring any training prior to making predictions. This means that their approach did not require any training or fine-tuning, while also being very efficient at the prediction stage. Even so, they achieved an accuracy of 96.84% for malware classification on the MalImg dataset, proving that more efficient approaches can still perform well for this task.

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

TODO: Introduce datasets

TODO: Insert summary table

#### Compiler detection
