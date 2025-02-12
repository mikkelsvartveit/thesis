# Methodology

## Research strategy

## Experimental setup

### Datasets

#### ISADetect

#### CpuReq

### Technical configuration

## Machine learning models

This research will primarily involve training, validating, and evaluating \ac{CNN} models using ISA characteristics such as endianness, word size, and instruction length as the target features. This subsection outlines our approach to data preprocessing, model architecture selection, and validation techniques.

### Data preprocessing

While most \ac{CNN} architectures are designed for image data, our datasets consist of compiled binary executables. Thus, how these are encoded into a format that can be consumed by a \ac{CNN} is a crucial part of our method. We aim to explore multiple approaches for image encoding.

The most primitive way of encoding a binary file is to treat each byte value as an integer whose value range from 0 to 255. For a classic two-dimensional \ac{CNN}, these integers will essentially be treated as pixel values, and the bytes form a grayscale image. Existing literature on malware detection from binary executables have explored this method of encoding. Kumari et al. [@Kumari2017] placed the byte values in a 2D array with a square aspect ratio, and resized all images to 150x150 pixels to ensure consistent input sizes. Other works have also applied similar techniques for encoding byte streams as square images, but with variations in image dimensions [@Prima2021] [@Hammad2022] [@Al-Masri2024]. Some researchers have used a similar approach, but fixing the image width while letting the height vary based on the file sizes [@El-Shafai2021] [@Alvee2021] [@Liang2021] [@Son2022]. Figure \ref{fig:byte-encoding} shows an example of a 9-byte sequence encoded as a 3x3 pixel grayscale image.

![Encoding bytes as a grayscale image. \label{fig:byte-encoding}](images/byte-encoding.svg)

Although \ac{CNN} is most frequently used with two-dimensional image data, one-dimensional \acp{CNN} are also common. Using the same byte-level encoding, we can convert a stream of bytes to a vector of integers between 0 and 255. This approach was also used by Yang et al. [@Yang2019] and Pizzolotto & Inoue [@Pizzolotto2021], who applied \acp{CNN} to detection of compiler optimization levels in binary executables.

### Model architectures

Hundreds of different \ac{CNN} architectures have been proposed in previous literature. LeNet-5, which is considered the first modern CNN architecture, has around 60 000 trainable parameters [@Lecun98]. Today, large-scale \ac{CNN} architectures such as VGG-16 often have over 100 million trainable parameters [@Simonyan2015].

Choosing a \ac{CNN} architecture is often a tradeoff between several factors:

- **Dataset size**: In general, more complex models require larger datasets. In cases where training data is limited, smaller architectures should be considered. Small dataset sizes combined with complex networks often lead to overfitting, meaning the model matches the training data so well that it fails to generalize to unseen data.

- **Training resources**: Larger models are more expensive to train in terms of computation power. Training deep learning models efficiently often requires use of powerful \acp{GPU}.

- **Inference resources**: Larger models do not only increase the cost of training, it also increases the cost of inference, i.e. making predictions using the trained model. Depending on where the model will be deployed, this may be a deciding factor.

#### Embeddings

### Validation

- LOGO-CV

## Evaluation

- CpuReq

### Baseline

- Andreassen
  - Clemens endianness heuristic
