# Results and Discussion

## Applications

Using CNN for analyzing binary machine code is not a novel idea. 18 of our 20 reviewed papers applies CNN to binary code for classifying malware. Two popular datasets are commonly used in this literature: Microsoft Malware Classification Challenge (MMCC) [SOURCE] and Malimg [SOURCE]. Table \ref{table:microsoft-results} and \ref{table:malimg-results} compares performance on these two datasets across the reviewed literature. The state-of-the-art performance is very high for both malware classification datasets. However, from what we have found, both datasets have large imbalances in the data amount types of malware, which causes model performance to flucuate across classes. The different papers address this to varying degree, and we consider this a limitation of the existing literature.

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

Two papers in our review used CNN for detecting compiler optimization levels from a compiled binary file. Knowledge of the compiler optimization level can be useful in areas such as vulnerability discovery. Yang et al. [SOURCE] created a dataset of ARM binaries compiled with GCC with four different optimization levels, and achieved an overall accuracy of 97.24% on their custom dataset, with precision for each class ranging from 96% to 98%. This was a significant improvement compared to previous non-CNN approaches regarding compiler analysis. Pizzolotto & Inoue [SOURCE] elaborated on this work by using binaries compiled across 7 different CPU architectures, as well as compiling with both GCC and Clang for the x86-64 and AArch64 architectures. They showed a 99.95% accuracy in distinguishing between GCC and Clang, while the optimization level accuracy varies from 92% to 98% depending on the CPU architecture.

## Encoding binary data

An evident challenge of using CNN for binary data is need for converting the bytes into a format that can be consumed by a CNN. This has been addressed through several approaches in the literature. The most straightforward method converts bytes into one-dimensional vectors of fixed length, typically ranging from 500 to 2000 elements [@Li2021] [@Rahul2017]. Some researchers enhance this approach with additional preprocessing steps such as word embedding [@Chaganti2022] [@Yang2019].

The most prevalent approach converts binaries into two-dimensional grayscale images, where bytes are represented as pixel values (0-255). These implementations either use fixed-size square images [@Kumari2017] [@Prima2021] or variable-height images with fixed widths [@Yang2018] [@El-Shafai2021]. More sophisticated techniques have emerged, including RGB image conversion [@Alam2024] and recurrence plot-based encoding [@Sartoli2020], with the latter showing improved classification consistency and accuracy compared to direct image conversion approaches.

## Transfer learning

## CNN variations
