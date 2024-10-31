# Results

<!-- ## Machine learning techniques for ISA detection -->

## CNN applied to binary code

Using CNN for analyzing binary machine code is not a novel idea. In this section, we will review the applications, datasets, and methods that have been previously explored within this domain.

### Applications

### Malware classification

Of the 21 included papers in the review, a total of 18 of them were papers on malware classification. These papers used CNN's to classify malware families from input malware binaries. Seeing as this is a comparetively commonly researched problem with regards to our research question **(RQ1?)**, we provide a comparison on different papers ability to classify malware from the two most commonly used datasets: Microsoft Malware Classification Challenge (MMCC) [@microsoftkaggle] and Malimg [@malimgpaper].

MMCC dataset contains malware binaries from 9 different malware families, while the Malimg dataset contains malware from 25 different families, at 21,741 and 9,339 malware samples respectivly [@microsoftkaggle] [@malimgpaper]. Across both datasets, 8 included papers used them in their research, where we generally see great results across all of the papers as seen in Tables \ref{table:microsoft-results} and \ref{table:malimg-results}. All papers used a one vs. all comparison when evaluating their models ability to classify malware families, and a macro average precision, recall and F1-scores for those that reported those metrics. The SREMIC model has the overall best results with 99.72% classification accuracy on the MMCC dataset and 99.93% on Malimg [@Alam2024], while the El-Shafai et al. paper from 2021 reports a 99.97% accuracy on the Malimg dataset at an apparent cost of a slightly worse f1-score [@El-Shafai2021].

However from what we have found, both datasets have large imbalances in the data amount types of malware and the different papers address this to verying degree. Rahul et al. [@Rahul2017], Kumari et al. [@Kumari2017], Khan et al. [@Khan2020], Sartoli et al. [@Sartoli2020], Son et al. [@Son2022] and Hammad et al. [@Hammad2022] all ignore the datasets imbalances, which could be taking into account when evaluating their performance. Yang et al. [@Yang2018] only classify between the two most represented malware families, while Liang et al. [@Liang2021], Cervantes et al. [@Garcia2019] and Al-Masri et al. [@Al-Masri2024] all use over- and/or undersampling. Li et al. [@Li2021] augmented their CNN with a XGBoost classifier, as a way of tackling the imbalance. We will touch more on this CNN variation. SREMIC [@Alam2024] and Bouchaib & Bouhorma [@Prima2021] generated aditional synthetic samples, where Bouchaib & Bouhorma used Synthetic Minority Oversampling Technique. SREMIC used a CycleGAN which in some cases generated 5 new images per malware file for the less represented malware families. Both SREMIC and Bouchaib & Bouhorma reports great results, but does not address how well their model would have performed without additional dataset generation. 


| Paper (year published)                  | Accuracy   | Precision | Recall | F1-score   |
| --------------------------------------- | ---------- | --------- | ------ | ---------- |
| Rahul et al. [@Rahul2017] (2017)        | 0.9491     | -         | -      | -          |
| Kumari et al. [@Kumari2017] (2017)      | 0.9707     | -         | -      | -          |
| Yang et al. [@Yang2018] (2018)          | 0.987      | -         | -      | -          |
| Khan et al. [@Khan2020] (2020)          | 0.9780     | 0.98      | 0.97   | 0.97       |
| Sartoli et al. [@Sartoli2020] (2020)    | 0.9680     | 0.9624    | 0.9616 | 0.9618     |
| Bouchaib & Bouhorma [@Prima2021] (2021) | 0.98       | 0.98      | 0.98   | 0.98       |
| Liang et al. [@Liang2021] (2021)        | 0.9592     | -         | -      | -          |
| SREMIC [@Alam2024] (2024)               | **0.9972** | -         | -      | **0.9988** |

Table: Microsoft Malware dataset classification performance. \label{table:microsoft-results}

| Paper (year published)                   | Accuracy   | Precision | Recall | F1-score   |
| ---------------------------------------- | ---------- | --------- | ------ | ---------- |
| Cervantes et al. [@Garcia2019] (2019)    | 0.9815     | -         | -      | -          |
| El-Shafai et al. [@El-Shafai2021] (2021) | **0.9997** | 0.9904    | 0.9901 | 0.9902     |
| Li et al. [@Li2021] (2021)               | 0.97       | -         | -      | -          |
| Son et al. [@Son2022] (2022)             | 0.97       | -         | -      | -          |
| Hammad et al. [@Hammad2022] (2022)       | 0.9684     | -         | -      | -          |
| S-DCNN [@Parihar2022] (2022)             | 0.9943     | 0.9944    | 0.9943 | 0.9943     |
| SREMIC [@Alam2024] (2024)                | 0.9993     | -         | -      | **0.9987** |
| Al-Masri et al. [@Al-Masri2024] (2024)   | 0.9989     | 0.9971    | 0.9984 | 0.9977     |

Table: Malimg dataset classification performance. \label{table:malimg-results}

#### Other datasets

#### Compiler detection

TODO

### Encoding binary data

Traditionally, CNN is widely used in visual tasks such as image classification, object detection, image segmentation, and computer vision [@cnn-survey]. Binary files, on the other hand, are just sequences of bytes, and exhibit no inherent multi-dimensionality in the same way as images or 3D objects do. Thus, a natural question to raise is: how do we convert a stream of bytes into a format that can be consumed by a CNN?

#### One-dimensional vector

Rahul et al. [@Rahul2017] proposed a method for converting binary data into a one-dimensional input vector. They preprocessed each file's binary content into a fixed-length vector of 128 decimal values. However, the paper does not elaborate on how this preprocessing is conducted.

Li et al. [@Li2021] also took a 1D encoding approach by converting each byte into a decimal value between 0 and 255. The bytes were then arranged as a one-dimensional vector and treated as a pixel value. Finally, the pixel vectors were compressed to a fixed size to ensure the input length stays the same across instances. They experimented with vector lengths of 500, 1000, and 2000, and their evaluation showed that while a longer input length increased accuracy, it also increased the computation time needed to train the model. They picked a vector length of 1000 as a trade-off between accuracy and computational efficiency.

Chaganti et al. [@Chaganti2022] used the ELF header to locate the entry point of each binary program. From this entry point, 2000 bytes were extracted. If there were less than 2000 bytes present after the entry point, the remaining bytes were padded with zero values. They then ran the bytes through an encryption cipher, performed base64 encoding of the encrypted bytes, and then used a word embedding layer before reaching the CNN.

Yang et al. [@Yang2019] and Pizzolotto et al. [@Pizzolotto2021] used CNN for detecting compiler optimization levels. Both converted the raw bytes into a vector of integers, and also included a word embedding layer before reaching the one-dimensional convolution blocks.

#### Two-dimensional grayscale image

The most common way to convert a binary file into a format interpretable by a CNN is to encode it as an image.

Kumari et al. [@Kumari2017] extracted each byte from the file and represented them as unsigned integers. Then, based on the file size, the integers were arranged in a 2D array with an aspect ratio close to 1:1. Each value was treated as a grayscale pixel with values between 0 and 255. Finally, the image was resized to 150x150 pixels to ensure identical input dimensions across instances. Prima & Bouhorma [@Prima2021], Hammad et al. [@Hammad2022], and Al-Masri et al. [@Al-Masri2024] used similar square image encodings, albeit with slightly different image sizes.

A variation of the square image is to use images of a fixed width, but variable length. Yang et al. [@Yang2018] fixed the image width to 512 pixels and let the height depend on the file size. El-Shafai et al. [@El-Shafai2021] took this a step further by defining a table of different width values based on the file size. Files over 1000 KB would get a width of 1024 pixels, files between 500 and 1000 KB would get a width of 768 pixels, and so on, with smaller files getting proportionally smaller widths. The latter approach was also used by Alvee et al. [@Alvee2021], Liang et al. [@Liang2021], and Son et al. [@Son2022].

#### Two-dimensional RGB image

SREMIC [@Alam2024] used RGB images for parts of their network. Similarly to the grayscale approaches discussed earlier, the bytes from the binary files were first converted to a set of vectors that form a 2D matrix. Then, they converted this matrix into a three-channel RGB image. Unfortunately, the paper does not explain how this conversion process was conducted.

#### Other approaches

While most existing literature uses a fairly straightforward image conversion pipeline, more sophisticated encoding approaches have been attempted.

Sartoli et al [@Sartoli2020] used an image conversion process based on recurrence plots. They viewed the binaries as a series of emissions from a stochastic process. From this, they generated 4092x4092 grayscale images using recurrence patterns. The images were resized to 64x64 pixels before training the CNN. They compared this to a direct image conversion approach, and found that the recurrence plots approach performed more consistently across classes, while also achieving a higher mean accuracy.

RansomShield [@Lachtar2023] utilized a Hilbert space-filling curve visualization of the binary file. They evaluated multiple CNN architectures, and found that the small and efficient LeNet model achieved a 99.7% accuracy on detecting Android ransomware from native machine instructions. LeNet outperformed deeper networks like VGG-16 while being up to 47 times more energy efficient.

### Transfer learning

Transfer learning is a machine learning technique where a model developed for one task is re-used for another task. Transfer learning is very useful for cases where there is not a lot of training data available, as well as in cases of limited computation power or time. Using a transfer learning approach can allow for deep networks despite these constraints.

Kumari et al. [@Kumari2017] experimented with two different transfer learning approaches in addition to training their own, significantly shallower, model from scratch. In the first approach, they used a pre-trained VGG-16 model and removed its fully connected layers at the end. Re-using only the convolutional blocks, they added and trained a small fully-connected network on top of the network. For the second approach, they also fine-tuned the last convolutional block before adding the same fully-connected network on top. Their evaluations showed that the pre-trained VGG-16 model with no fine-tuning performed the best among their three approaches, proving that using pre-trained CNN models optimized for images also can perform well for other applications. Prima et al. [@Prima2021] also did transfer learning by using VGG-16 with a fully connected block at the end. They found that the transfer learning model achieved the same performance as their from-scratch CNN, but a limitation of the paper is that they do not outline the architecture of the model they trained from scratch.

El-Shafai et al. [@El-Shafai2021] compared the performance of eight different pre-trained CNN models. They used transfer learning with fine-tuning, and found the best performing model to be VGG-16. It achieved a striking 99.97% accuracy for malware classification on the MalImg dataset, while reducing the number of training parameters by 99.92% compared to training VGG-16 from scratch.

Hammad et al. [@Hammad2022] used a pre-trained GoogLeNet model, which is designed with computational efficiency and memory footprint in mind. Compared to VGG-16's 138 million, GoogLeNet only needs 4 million parameters. To make predictions, the authors used a basic k-nearest neighbors (KNN) classifier on top. A notable feature of KNN models is that they do not requiring any training prior to making predictions. This means that their approach did not require any training or fine-tuning, while also being very efficient at the prediction stage. Even so, they achieved an accuracy of 96.84% for malware classification on the MalImg dataset, proving that more efficient approaches can still perform well for this task.

### CNN variations

The conventional architecture for CNN includes convolution layers, pooling layers, activation layers, and fully-connected layers. However, prior research has also explored alternative or augmented CNN architectures for these applications.

Li et al. [@Li2021] evaluated the use of an **XGBoost** (eXtreme Gradient Boosting) classifier on top of the CNN. They essentially replaced the Softmax activation layer at the end with an XGBoost classifier, using the CNN as a feature extractor and XGBoost for the final classification. The authors claimed that by using XGBoost, they could combat overfitting and low accuracy in cases of unbalanced data. Their evaluation showed that this was indeed the case. Overall accuracy increased from 95% to 97%. More importantly, the accuracy and F1-scores for particular underrepresented classes saw a dramatic improvement.

Liang et al. [@Liang2021] invented a custom architecture dubbed **MFF-CNN** (Multi-resolution Feature Fusion Convolutional Neural Network). Here, they begin with creating three different resolutions of each image. They started with 112x112 images, and then created downscaled 56x56 and 28x28 version using max-pooling. These three versions went through parallel CNNs with a Spatial Pyramid Pooling (SPP) layer at the end. SPP ensures that the output of the CNN is of a fixed size, even if the input size varies. The result of this was three feature vectors of size 1050x1, one from each resolution. The authors then used a feature fusion step where a weighted average method combined the three vectors. The evaluation showed that this approach was particularly effective for distinguishing similar malware families in the dataset, which was an improvement over previous literature.

TODO:

- S-DCNN
- Spatial CNN
- Bi-GRU-CNN
- Dual CNN
