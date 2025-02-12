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

While most \ac{CNN} architectures are designed for image data, our datasets consist of compiled binary executables. Thus, how these are encoded into a format that can be consumed by a \ac{CNN} is a crucial part of our method. In our experiments, we use two different approaches for image encoding.

#### Two-dimensional byte-level encoding

We treat each byte value as an integer whose value range from 0 to 255. The values are placed in a two-dimensional array of a predetermined size. If the file is larger than the predetermined size, only the first bytes are used. If the file is smaller than the predetermined size, the remaining bytes are padded with zero values.

When applying two-dimensional \ac{CNN} on 2D grids of this format, the byte values will essentially be treated as pixel values, where the byte sequence forms a grayscale image. Figure \ref{fig:byte-encoding} shows an example of a 9-byte sequence encoded as a 3x3 pixel grayscale image.

![Encoding bytes as a grayscale image. \label{fig:byte-encoding}](images/byte-encoding.svg)

This approach was chosen based on previous literature which successfully classified malware from binary executables using \acp{CNN} [@Kumari2017] [@Prima2021] [@Hammad2022] [@Al-Masri2024] [@El-Shafai2021] [@Alvee2021] [@Liang2021] [@Son2022].

#### One-dimensional byte-level encoding

Similar to the 2D approach, we treat each byte as an integer. The values are placed in a one-dimensional array of a predetermined size. If the file is larger than the predetermined size, only the first bytes are used. If the file is smaller than the predetermined size, the remaining bytes are padded with zero values.

This approach was chosen based on previous literature which successfully detected compiler optimization levels in binary executables using 1D \acp{CNN} [@Yang2019] [@Pizzolotto2021].

### Model architectures

#### Embeddings

### Validation

- LOGO-CV

## Evaluation

- CpuReq

### Baseline

- Andreassen
  - Clemens endianness heuristic
