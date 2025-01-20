# Results and Discussion

## Applications

Out of our 20 reviewed articles, 18 apply CNN to binary code for the purpose of classifying malware. Two popular datasets are commonly used in this literature: Microsoft Malware Classification Challenge (MMCC) [@microsoftkaggle] and Malimg [@malimgpaper]. Table \ref{table:microsoft-results} and \ref{table:malimg-results} compare performance on these two datasets across the reviewed literature. The state-of-the-art performance is very high for both malware classification datasets. However, from what we have found, both datasets have large imbalances in the data amount across types of malware, which causes model performance to fluctuate across classes. The different articles address this issue to varying degrees, and we consider this a limitation of the existing literature.

| Article (year published)             | Accuracy   | Precision  | Recall     | F1-score   |
| ------------------------------------ | ---------- | ---------- | ---------- | ---------- |
| Rahul et al. [@Rahul2017] (2017)     | 0.9491     | -          | -          | -          |
| Kumari et al. [@Kumari2017] (2017)   | 0.9707     | -          | -          | -          |
| Yang et al. [@Yang2018] (2018)       | 0.987      | -          | -          | -          |
| Khan et al. [@Khan2020] (2020)       | 0.9780     | 0.98       | 0.97       | 0.97       |
| Sartoli et al. [@Sartoli2020] (2020) | 0.9680     | 0.9624     | 0.9616     | 0.9618     |
| Prima & Bouhorma [@Prima2021] (2021) | 0.98       | 0.98       | 0.98       | 0.98       |
| Liang et al. [@Liang2021] (2021)     | 0.9592     | -          | -          | -          |
| SREMIC [@Alam2024] (2024)            | **0.9972** | **0.9993** | **0.9971** | **0.9988** |

Table: Microsoft Malware dataset classification performance. Bolded values indicate state-of-the-art performance.\label{table:microsoft-results}

| Article (year published)                 | Accuracy   | Precision  | Recall     | F1-score   |
| ---------------------------------------- | ---------- | ---------- | ---------- | ---------- |
| Cervantes et al. [@Garcia2019] (2019)    | 0.9815     | -          | -          | -          |
| El-Shafai et al. [@El-Shafai2021] (2021) | **0.9997** | 0.9904     | 0.9901     | 0.9902     |
| Li et al. [@Li2021] (2021)               | 0.97       | -          | -          | -          |
| Son et al. [@Son2022] (2022)             | 0.97       | -          | -          | -          |
| Hammad et al. [@Hammad2022] (2022)       | 0.9684     | -          | -          | -          |
| S-DCNN [@Parihar2022] (2022)             | 0.9943     | 0.9944     | 0.9943     | 0.9943     |
| SREMIC [@Alam2024] (2024)                | 0.9993     | **0.9992** | **0.9987** | **0.9987** |
| Al-Masri et al. [@Al-Masri2024] (2024)   | 0.9989     | 0.9971     | 0.9984     | 0.9977     |

Table: Malimg dataset classification performance. Bolded values indicate state-of-the-art performance.\label{table:malimg-results}

Two articles in our review used CNN for detecting compiler optimization levels from a compiled binary file. Knowledge of the compiler optimization level can be useful in areas such as vulnerability discovery. Yang et al. [@Yang2019] created a dataset of ARM binaries compiled with GCC with four different optimization levels, and achieved an overall accuracy of 97.24% on their custom dataset, with precision for each class ranging from 96% to 98%. This was a significant improvement compared to previous non-CNN approaches regarding compiler analysis. Pizzolotto & Inoue [@Pizzolotto2021] elaborated on this work by using binaries compiled across 7 different CPU architectures, as well as compiling with both GCC and Clang for the x86-64 and AArch64 architectures. They showed a 99.95% accuracy in distinguishing between GCC and Clang, while the optimization level accuracy varies from 92% to 98% depending on the CPU architecture.

Based on these performance metrics in both cases, CNN show great potential for binary code analysis. The consistently high accuracies across different CNN architectures shows promise in future applications and other usecases. However, several limitations should be noted. Both malware datasets are quite imbalanced, which casts doubts on effectiveness in real-world applications. The exceptionally high accuracies and F1-scores on the Malimg dataset also raises questions about the quality of the dataset, and a potential need for further evaluation benchmarks. There are also large inconsistencies in the quality of performance metrics reported from the papers, making in-depth comparisons difficult. Nevertheless, recent advances in addressing data imbalance through augmentation techniques, as demonstrated by SREMIC [@Alam2024], combined with successful applications in both malware classification and compiler optimization detection, provide strong evidence for CNNs' versatility in binary code analysis.

## Encoding binary data

An evident challenge of using CNN for binary data is need for converting the bytes into a format that can be consumed by a CNN. This has been addressed through several approaches in the literature. The most straightforward method converts bytes into one-dimensional vectors of fixed length, typically ranging from 500 to 2000 elements [@Li2021] [@Rahul2017]. Some researchers enhance this approach with additional preprocessing steps such as word embedding [@Chaganti2022] [@Yang2019] [@Pizzolotto2021].

The most prevalent approach converts binaries into two-dimensional grayscale images, where bytes are represented as pixel values (0-255). These implementations either use fixed-size square images [@Kumari2017] [@Prima2021] [@Hammad2022] [@Al-Masri2024] or variable-height images with fixed widths [@Yang2018] [@El-Shafai2021] [@Alvee2021] [@Liang2021] [@Son2022]. More sophisticated techniques have emerged, including RGB image conversion [@Alam2024] and recurrence plot-based encoding [@Sartoli2020], with the latter showing improved classification consistency and accuracy compared to direct image conversion approaches.

## Transfer learning

Transfer learning, where CNN models pre-trained on one task are adapted for another, offers significant advantages when dealing with limited data, computational resources, or time constraints. Prior research on CNN for binary analysis has demonstrated its effectiveness in malware classification tasks. Kumari et al. [@Kumari2017] and Prima & Bouhorma [@Prima2021] showed that pre-trained VGG-16 models, modified with new fully-connected layers, could match or exceed the performance of models trained from scratch. El-Shafai et al.'s [@El-Shafai2021] comprehensive comparison of eight pre-trained CNNs found that VGG-16 achieved 99.97% accuracy in malware classification while reducing training parameters by 99.92% compared to training the same network from scratch. Notably, Hammad et al. [@Hammad2022] demonstrated that even more efficient architectures like GoogLeNet, combined with a KNN classifier, could achieve 96.84% accuracy without requiring additional training, showing the versatility of transfer learning approaches.

## CNN variations

Some reviewed primary studies have explored several alternative CNN architectures for malware classification. Li et al. replaced the traditional Softmax layer with XGBoost, using the CNN as a feature extractor and XGBoost for final classification [@Li2021]. In their evaluation, this improved overall accuracy from 95% to 97% and significantly enhanced performance on underrepresented classes. Liang et al. developed MFF-CNN (Multi-resolution Feature Fusion CNN), processing multiple image resolutions through parallel CNNs with Spatial Pyramid Pooling [@Liang2021]. Their weighted feature fusion approach proved particularly effective at distinguishing similar malware families. The S-DCNN architecture by Parihar et al. combined pre-trained ResNet50, Xception, and EfficientNet-B4 models, with features concatenated and processed through a fully-connected neural network [@Parihar2022]. This ensemble approach achieved state-of-the-art performance with a 99.43% F1-score on the Malimg dataset.
