# Discussion

## Overview of key findings

In this section, we will briefly go over the main findings in our experiments. We will also highlight the potential implications of these findings related to our research questions by indicating further points of discussion for later sections.

### K-fold cross validation and evaluation strategies

In the initial K-fold cross validation experiments on ISAdetect we observed that all models performed similarly well, with next to 100% classification accuracy on both endianness and instruction width type. This result was also achieved with very little variance between runs, and points to the fact that already seen architectures across the training and test set lets the models excel at fitting to endianness and instruction width. The incredibly high accuracy feeds our suspicion that the models can quickly fit to architectural features that are not inherently tied to endianness and instruction width. We believe it was correct in our assessment that this needed to be investigated further, and that \ac{LOGO CV} and train-testing on different datasets is the best way of evaluating the \acp{CNN} general ability to detect \ac{ISA} features.

In the \ac{LOGO CV} on ISAdetect runs and the evaluation strategies using multiple datasets we see more varied and interesting results compared to K-fold. In terms of raw classification performance, individual model performance varies quite a lot depending on the experimental suite and datasets. On the \ac{LOGO CV} suite with ISAdetect, we are able to achieve 90.3% average accuracy on endianness detection and 88.0% accuracy on instruction width detection. Both of these scores were achieved with the Simple1d-E model. The performance on the other suites are not quite as convincing, as on the suites testing on CpuRec we see higher variability between runs with lower accuracies, and lower overall accuracy on the BuildCross suites.

### Endianness detection

With endianness detection we see the clear performance differences between embedding and non-embedding models. The embedding models perform better in 3 out of 4 test suites, with a clear gap in performance on \ac{LOGO CV} ISAdetect. The only exception is ISAdetect-Buildcross where both 1d and 2d non-embedding versions beat out their embedding counterparts, with a lot less variance in the reported average accuracy as well. When comparing model dimensionality and complexity when testing on CpuRec based on the 95% confidence interval, we see no statistical significant difference between the 1d, 2d and ResNet50. Still, the large performance difference on \ac{LOGO CV} ISAdetect and training on ISAdetect + Buildcross and testing on CpuRec experiments indicates that the embedding models are better suited for endianness detection, where the results carry significant differences and test-suites are more comprehensive in terms of evaluation strategy and dataset size.

### Instruction width type detection

For instruction width type detection, it is even less clear which variations of models perform best. In the ISAdetect-CpuRec suite the performance is not great, with results comparable to random guesswork hovering between 53.2% and 60.1% average accuracy across all. Just like with endianness detection, the two best performing suites are \ac{LOGO CV} ISAdetect and ISAdetect+BuildCross on CpuRec. We also see here that embedding models perform better, although with less performance differences across model variations. While it seems that embedding models are better suited for instruction width detection, the performance differences are not as pronounced as with endianness detection.

### Addition of BuildCross

We believe that the addition of the BuildCross dataset has had a positive impact on the overall quality and robustness of our testing methodology, as shown in the results from both ISAdetect-BuildCross and ISAdetect+BuildCross on CpuRec experiments. Our ISAdetect-BuildCross performance shows comparable results to ISAdetect-CpuRec on instruction width detection, validating our efforts in creating a dataset with more exotic and diverse datasets. However, an interesting anomaly appears with endianness detection in the same suite, as the non-embedding models outperform their embedding counterparts with remarkably low variance. This unexpected pattern contrasts the other experimental suites. There is some indication of similar behavior, although less pronounced, in the instruction width detection experiments on the same suite.

While ISAdetect-BuildCross showed some interesting patterns, the most significant impact in terms of overall accuracy appeared when combining ISAdetect and BuildCross. The addition of BuildCross significantly improved performance on the CpuRec testing suite, particularly for instruction width type detection. In this case, performance jumped from around 50-60% to above 80% for the top performing models. This indicates that the addition of more diverse and exotic architectures in the training data has a positive impact on the overall performance of our models.

### Simple vs complex models

It is worth noting that the higher complexity of ResNet architectures did not translate to significant performance gains in our testing. Despite having significantly more parameters and deeper architecture, ResNet50 and ResNet50-E typically performed on par with or worse than simpler models. Training loss patterns across different experimental suites suggest that the features related to endianness and instruction width type may not be complex enough to require the additional representational power that ResNet provides<!-- TODO: training loss -->. This observation holds true even when considering that ResNet models were given a larger input window of 1024 bytes compared to the 512 bytes used by simpler models. The result suggests that lightweight models may be preferable for these specific ISA feature detection tasks, offering comparable accuracy with lower computational requirements.

## Model architecture performance analysis

<!--
TODO:

- Visualize some grayscale images
- Learning rate converges fast relative to the amount of data we have, suggest that it is fitting to something

-->

### Impact of embedding layers

In most of our experiments, we see that the model architectures that employ an embedding layer as the first layer of the model perform significantly better than their non-embedding counterparts. This is a key finding, and aligns with our hypothesis that embedding techniques may improve performance for \ac{CNN} models due to the categorical nature of binary code.

Consider this simple instruction for the Intel 8080 instruction set:

```assembly
ADI 25;
```

It uses the `ADI` opcode, which indicates an addition with an immediate value. It sums the content of the accumulator register and the immediate value, and saves the result to the accumulator register. We can examine what this looks like when assembled to a 16-bit binary instruction:

$$
\underbrace{1100\ 0110}_{Opcode} \ \ \underbrace{0001\ 1001}_{Immediate\ value}
$$

The first byte contains the operation code. While operation codes are represented as numbers in the executable code, there is no semantic meaning to this number. It is actually a discrete, categorical piece of data that have no semantic relationship to bytes of close values such as $1100\ 0101$ and $1100\ 0111$.

Intuitively, an operation that is semantically similar to `ADI` (Add Immediate) is `SUI` (Sub Immediate). It performs the same operation, but subtracts the immediate value from the accumulator instead of adding it. The opcode for `SUI` is $1101\ 0110$. Converting this to base 10, the numbers used to represent the `ADI` and `SUI` instructions are 198 and 214. These values themselves do not properly represent the close semantic relationship between the operations.

However, introducing an embedding layer in the model makes it capable of identifying and learning semantic relationships such as this by converting each byte value into a continuous vector. Bytes with close semantic relationships would be represented as similar vectors. While this is a very simple example, converting categorical data into semantic-capturing vectors is a powerful technique that often results in superior performance when training and testing deep learning models on categorical input.

### Model complexity

A clear trend in our results is that the large ResNet models do not perform better than the smaller and simpler \ac{CNN} architectures, and in many cases they actually perform worse than the smaller models. A possible explanation for this is that the ResNet models' high representational power might overfit on the training data. This typically happens when the size or diversity of the training data is limited.

While we consider our data quantity to be sufficient, there are reasons to believe that the diversity of the data is not high enough to avoid overfitting when training larger models. This claim is also supported by the fact that every model we trained converged rather quickly, almost always after just one or two epochs. The limited representational power of the smaller models may actually be beneficial in our case, since they are forced to learn simpler and more obvious patterns instead of picking up on what might effectively be random noise in the training data.

TODO: bar chart showing the parameter count of each model

### CNN dimensionality

While most applications of \acp{CNN}, such as image analysis, use two-dimensional convolution layers, we also included one-dimensional models in our experiments. Prior to running our experiments, we hypothesized that two-dimensional \acp{CNN} might perform better than the one-dimensional ones due to the repeating patterns of fixed-width instruction sets. We also chose the 32x16 dimensions for the same reason, considering that many \acp{ISA} use 32-bit wide instructions.

Our results indicate that for detecting endianness, the two-dimensional models generally do not show an advantage over the one-dimensional counterparts. It is likely that the models do not rely on repeating patterns for detecting endianness, since endianness fundamentally operates on an individual byte organization level.

For detecting instruction width type, the two-dimensional models do perform as well or better than the one-dimensional models for experiments that use CpuRec or BuildCross as the test set. However, for \ac{LOGO CV} on ISADetect, the one-dimensional models still perform slightly better.

The relationship between model dimensionality and performance appears to be influenced by both the specific architectural feature being detected and the diversity of the training/testing datasets. This indicates that optimal \ac{CNN} dimensionality for binary code analysis may be feature-dependent, rather than universally favoring a particular approach.

### Variance in model performance

When training a deep learning model, several components use pseudo-randomness:

- **Weight initialization:** The trainable model parameters are initialized with random values before training starts. This is usually preferred over starting with all parameters set to zero.
- **Mini-batch sampling:** For each training iteration, a random subset of the training data is used to compute the next weight update.
- **Dropout:** A random set of neurons in each layer is set to zero during training.

To control and reproduce these pseudo-random elements, one can specify a seed. Setting a seed guarantees that the pseudo-random behavior can be reproduced. When training our models, we train and test it multiple times using different seeds. This allows us to compare the accuracy between different random initializations.

Generally, we observe a very high variance between different runs due to differences in randomness. This indicates that the training process of the model is unstable, where the performance on an unseen test set varies greatly even if the training loss quickly converges to zero. For instance, the best-performing model for endianness detection (_Simple1d-E_), when evaluating with \ac{LOGO CV} on ISADetect, shows a standard deviation of up to 28 percentage points for certain \acp{ISA} when comparing the accuracy across different random seeds (see \autoref{table:logo-endianness-results}). This happens even though we take precautions such as using low learning rates and regularizing the models with dropout.

This is common behavior when the size of the training dataset is limited. While we consider our training dataset to be large and comprehensive, the model variability strengthens our suspicion that the dataset is too homogeneous for training deep neural networks in an optimal way. Another factor that might cause these results is outliers in the data. Random initialization might make models more or less sensitive to outliers in the training data.

### Architecture Outliers

<!-- m68k LOGO CV,  -->

## Model generalizability

A key objective of our models is to be able to generalize to \acp{ISA} that were not seen during training. This section analyzes the generalizability of our models and how our experiments support this objective.

### Leave-one-group-out cross validation

We use \ac{LOGO CV} as our cross validation method for the ISAdetect dataset. In contrast to standard K-fold cross validation, \ac{LOGO CV} tests how the model performs on a previously unseen group. This is a more realistic scenario for testing generalizability, since it simulates the real-world scenario where a model is deployed to a new \ac{ISA} that was not seen during training.

To showcase the effectiveness of \ac{LOGO CV}, we compare the accuracy of our models when evaluated on \ac{LOGO CV} to the accuracy when evaluated with a standard train/test split using 80% of the data for training and 20% for testing. The results of the latter approach on the best-performing _Simple1d_ model is shown in \autoref{fig:isadetect-traintest-accuracy-by-isa}. We observe extreme performance, achieving an average accuracy of 99.99%. In contrast, the same setup using \ac{LOGO CV} gave an accuracy of 89.7%, as observed in \autoref{training-and-testing-on-isadetect}. It is clear that evaluating on the same \acp{ISA} as the ones present in the training data results in performance that is artificially high when the overall objective is to evaluate generalizability to unseen \acp{ISA}.

TODO: Change this image, it is wrong

![Evaluating the _Simple1d_ model architecture using a standard train/test split on the ISAdetect dataset \label{fig:isadetect-traintest-accuracy-by-isa}](./images/isadetect-traintest-accuracy-by-isa.png)

### Testing on other datasets

For evaluating the generalizability beyond the 23 \acp{ISA} present in the ISAdetect dataset, we use the CpuRec dataset as well as BuildCross, the custom dataset we developed for this thesis. These datasets provide a more diverse set of \acp{ISA} than the ISAdetect dataset. In particular, CpuRec contains binaries from 76 different \acp{ISA}, while BuildCross contains binaries from 40 different \acp{ISA}.

#### CpuRec

We observe that models trained on ISAdetect do not generalize well to the CpuRec dataset. While certain models appears to perform well, it is important to note that there is an overlap between the \acp{ISA} present in the ISAdetect and CpuRec datasets. \autoref{fig:dataset-isa-overlap} illustrates this. Out of the 76 \acp{ISA} present in CpuRec, 16 of them are also present in ISAdetect.

![Venn diagram illustrating the overlap of \acp{ISA} present in the ISAdetect, CpuRec and BuildCross datasets \label{fig:dataset-isa-overlap}](./images/discussion/dataset-isa-overlap.svg)

This overlap of \acp{ISA} between the datasets is a limitation of our experiments, as it may lead to models memorizing specific \acp{ISA} characteristics rather than learning generalizable features. However, we can mitigate this by excluding the \acp{ISA} present in ISAdetect from the CpuRec dataset, and observe the performance using only the non-overlapping \acp{ISA}. Endianness classification performance after excluding the \acp{ISA} present in ISAdetect from the test set is shown in \autoref{fig:cpurec-endianness-by-model-exclude-overlap}. We observe that the model with the highest accuracy is now Simple2d, achieving an accuracy of 74.7%, down from Simple1d-E's 81.0% when evaluated on the entire CpuRec dataset. For instruction width type classification, the effect of removing overlapping \acp{ISA} is even more pronounced. \autoref{fig:cpurec-instructionwidthtype-by-model-exclude-overlap} shows the instruction width classification performance after excluding the \acp{ISA} present in ISAdetect from the test set. Here, the best-performing model only achieves an accuracy of 44.9%, which is worse than what a dummy model that always predicts the most common class would achieve.

![Endianness classification performance on the CpuRec dataset after excluding the \acp{ISA} present in ISAdetect \label{fig:cpurec-endianness-by-model-exclude-overlap}](./images/discussion/cpurec-endianness-by-model-exclude-overlap.svg)

![Instruction width classification performance on the CpuRec dataset after excluding the \acp{ISA} present in ISAdetect \label{fig:cpurec-instructionwidthtype-by-model-exclude-overlap}](./images/discussion/cpurec-instructionwidthtype-by-model-exclude-overlap.svg)

We identify several potential reasons for the poor generalizability of our ISAdetect-trained models:

- The diversity of the ISAdetect dataset used for training is quite limited. CpuRec contains 76 different \acp{ISA}, while ISAdetect only contains 23. In addition, the ISAdetect dataset is more homogeneous, with all \acp{ISA} being supported compile targets for recent versions of the Debian Linux distribution. CpuRec, on the other hand, was developed by manually cross-compiling source code to a very diverse set of \acp{ISA}. By inspecting the \ac{ISA} features in the two dataset, we can for instance observe that while all \acp{ISA} in ISAdetect have 32 or 64 bit word sizes, CpuRec also contains several \acp{ISA} with 8 and 16 bit word sizes.

- The CpuRec dataset only contains a single binary file per \ac{ISA}. This is a significant limitation of the dataset that makes our results less conclusive and more sensitive to anomalies in the specific binary used for each \ac{ISA}.

- Due to the nature of deep learning, it is possible that the \ac{CNN} models are picking up on \ac{ISA}-specific patterns that are not inherently related to the endianness or instruction width. This is a common problem in deep learning, and is known as overfitting to the training data. Since it is difficult to interpret the inner workings of \ac{CNN} models, we can only speculate whether this is the case. However, the high accuracies observed when running K-fold cross validation on the ISAdetect dataset do support the claim that the models are easy to fit to full \acp{ISA} compared to fitting them to specific \ac{ISA} characteristics.

As noted in the results chapter, our findings show that augmenting the training data with BuildCross does not improve the generalizability of endianness detection. However, we do see indications that the instruction width type classification task benefits from augmenting the training data with BuildCross. To make this a fair comparison, we must note that training on BuildCross results in more \acp{ISA} overlap between the training and test datasets, as compared to training on ISAdetect only. To emphasize that the performance is actually better on unseen \acp{ISA}, we can examine the results when excluding both the \acp{ISA} present in ISAdetect and BuildCross from the test set. \autoref{fig:combined-instructionwidthtype-by-model-exclude-overlap} illustrates this. Compared to \autoref{fig:cpurec-instructionwidthtype-by-model-exclude-overlap}, we see significant performance improvements across all model architectures, indicating that the inclusion of a more diverse training dataset does improve the generalizability of instruction width type classification. (TODO: reason more about why this is the case)

![Instruction width classification performance on the CpuRec dataset after excluding the \acp{ISA} present in ISAdetect and BuildCross\label{fig:combined-instructionwidthtype-by-model-exclude-overlap}](./images/discussion/combined-instructionwidthtype-by-model-exclude-overlap.svg)

#### BuildCross

BuildCross is the dataset we developed specifically for this thesis, containing binaries from 40 different \acp{ISA}. In contrast to the other experiments, we observe that the non-embedding models perform better when evaluated on this dataset. Particularly, the best-performing model for endianness classification is Simple1d, achieving an accuracy of 71.3%. For instruction width type classification, the best-performing model is Simple2d, achieving an accuracy of 69.6%.

An advantage of the BuildCross dataset compared to the CpuRec dataset is that there is little \acp{ISA} overlap with the training dataset (ISAdetect). This reduces the risk of the performance numbers showing up as artificially high due to the models memorizing specific \acp{ISA} characteristics rather than learning generalizable features.

We note that while generalizability for the endianness classification task seem similar between the CpuRec and BuildCross datasets, the instruction width type classification task shows a clear improvement when evaluated on the BuildCross dataset. (TODO reason more about why this is the case.)

## Comparison with prior literature

### Andreassen and Morrison

<!--
(Stian)
- Prior litterature introduction: andreassen
  - Same supervisor as us, we got referenced this thesis from him (morrison)
  - Comment on which test suites are the same, what he did diferently (labling, included architectures)
  - Endiannes:
    - LOGO CV ISAdetectCode only:
      - 92.0% bigram, 91.7% EndiannessSignature. Rand forest
    - Isadetect code CpuRec:
      - 86.3% LogisticRegression and RandomForest bigram. 82,4% SVC EndiannessSignature
  - Fixed vs Variable width
    - LogoCv CpuRec:
      - 88.4% RandomForest with autocorrelation
      - Not comparable since we did not have enough training data to consider this an experimental suite for inclusion
  - inherent differences in approach
    - He did feature engineering, we tried to do it with CNNs and automatically extract features
    - CNNs and deep learning require more data, and we are not able to train on the same amount of data as he did for some of his suites
    - We have different labling of datasets, and we are not able to compare the results directly
  - Critique of Andreassen
    - Reproducton of his results fell outside the scope of this thesis
    - Doesn’t exclude previously seen architectures when testing on CPURec
    - Lacking a lot of labels and mislabeling certain things
    - Does not appear to do multiple runs with different seeds, might be a problem with random initialization, while likely less of a problem for simpler ml models? TODO check this
    -->

In our search for related work documented in \autoref{related-work}, the thesis by Andreassen and Morrison [@Andreassen_Morrison_2024] stands out as the only other identified research that specifically addresses the problem of detecting individual \ac{ISA} features from unknown binary code. This work was supervised by Donn Morrison, who is also the supervisor of the current thesis and who recommended we review this research. For clarity in the following discussion, we will refer to this paper as "Andreassen's work," acknowledging Morrison's supervisory role in that project. The thesis uses a similar approaches and datasets, but with different feature extraction methods. Andreassen uses explicit feature engineering with classical machine learning classifiers for targeting the different \ac{ISA} features, while we used \acp{CNN} to automatically extract features from the binary code. While the thesis also targets endianness and instruction width type detection, he also includes the third target feature of detecting instruction width size of fixed-width architectures.

The thesis uses some of the same experimental suites as we do, with the same datasets and evaluation strategies. \ac{LOGO CV} was a key part of in all of his suites, in addition to training on ISAdetect and testing on CpuRec, \ac{LOGO CV} with CpuRec and training on CpuRec testing on ISAdetect. We will compare the results of our models with the results of Andreassen's models where applicable. However, there are some key differences in the labeling of datasets and the architectures used for training and testing, which makes a completely accurate and direct comparison difficult. In the next subsections we present our interpretation of a direct performance comparison on endianness and instruction width type classification, before discussing the potentially impactful differences in our approaches and addressing comparison issues.

#### Endianness

There are two experimental suites set up in [@Andreassen_Morrison_2024] that are comparable to our results. The first suite is \ac{LOGO CV} on ISAdetect on code sections, where he achieved an accuracy of 92.0% using a Random Forest classifier with bigram features, and 91.7% using a Random Forest classifier with EndiannessSignature features. In comparison, we achieve an accuracy of 90.3% using the Simple1d-E model on the same dataset. Andreassen's bigram feature consists of counting up all combinations of two byte-pairs in each program, resulting in a histogram of $256 \cdot 256 = 65,536$ input features. The EndiannessSignature was originally developed by [@Clemens2015] and is a feature vector of the counts of only 4 bigrams, 0xfffe, 0xfeff, 0x0001, and 0x0100. Increment and decrement by one are common operations in computer programs, and these bigrams can capture this difference across the two endianness types<!-- TODO write about in related work, maybe not needed here -->. Our results are a bit lower than Andreassen's on average, but within the 95% confidence interval of ±2.0% for our model. The other comparable suite is training on ISAdetect code sections and testing on CpuRec. Andreassen achieved an accuracy of 86.3% using a Random Forest and Logistic Regression classifier with bigram features. In comparison, we were only able to achieve an accuracy of 75.4%, 76.3% and 76.4% using the Simple1d-E, Simple2d-E and ResNet50-E models respectively in our testing. This is a relatively large difference in performance. One advantage of Andreassen's feature extraction methods is the use of entire binary files, as both the bigram-based features are able to gather statistical information from the entire file. \acp{CNN} automatic feature extraction are limited to the input window of the model. This is also an advantage of \acp{CNN}, as they might be better suited for smaller samples code sections.

#### Instruction width type

#### Differences in approach and critique

Andreassen does not list statistical evidence for the accuracy of his models, and labeling differences makes a statistically significant comparison impossible.

### Instruction Set Architecture Detection

## Dataset quality assessment

### ISAdetect dataset

The ISAdetect is our main dataset for training machine learning models. It contains 23 different \acp{ISA}, with each architecture containing between 2800 and 6000 binary program samples [@Kairajarvi2020] [@Clemens2015] per architecture. We train our models on the code sections of these binary program samples, and exclude the code sections that are smaller than 1024 bytes. We do not perform file splitting on the ISAdetect dataset to augment the amount of training data, as preliminary results revealed that this did not improve model performance or generalizability. Taking this into consideration, \autoref{table:isadetect-samples} shows the number of samples per \ac{ISA} in ISAdetect. This averages to 4,086 samples per \ac{ISA}.

Table: Number of samples per \ac{ISA} in ISAdetect. \label{table:isadetect-samples}

| \ac{ISA}   | No. samples |
| ---------- | ----------: |
| s390       |       5,118 |
| sparc      |       4,923 |
| armhf      |       3,674 |
| i386       |       4,484 |
| arm64      |       3,518 |
| armel      |       3,814 |
| sh4        |       5,854 |
| amd64      |       4,059 |
| riscv64    |       4,285 |
| mipsel     |       3,693 |
| s390x      |       3,511 |
| powerpc    |       3,618 |
| mips       |       3,547 |
| m68k       |       4,313 |
| ppc64el    |       3,521 |
| x32        |       4,059 |
| hppa       |       4,830 |
| powerpcspe |       3,922 |
| alpha      |       3,952 |
| sparc64    |       3,205 |
| mips64el   |       4,280 |
| ppc64      |       2,822 |
| ia64       |       4,983 |
| **Total**  |  **93,985** |

We can examine the number of samples for each class for both of our target features, endianness and instruction width type. \autoref{table:isadetect-endianness-samples-per-class} and \autoref{table:isadetect-instructionwidthtype-samples-per-class} show the number of samples per class for endianness and instruction width type, respectively. We can see that for both target features, there are more than 30,000 training instances per class. This should be sufficient for training even the most complex \ac{CNN} models.

Table: Number of samples per class for endianness in ISAdetect. \label{table:isadetect-endianness-samples-per-class}

| Endianness | No. samples | Percentage |
| ---------- | ----------: | ---------: |
| little     |      54,176 |     57.64% |
| big        |      39,809 |     42.36% |

Table: Number of samples per class for instruction width type in ISAdetect. \label{table:isadetect-instructionwidthtype-samples-per-class}

| Instruction width type | No. samples | Percentage |
| ---------------------- | ----------: | ---------: |
| fixed                  |      63,458 |     67.52% |
| variable               |      30,527 |     32.48% |

We also observe that there is some class imbalance in the dataset, particularly for instruction width type, where more than two thirds of the binaries have fixed-width instructions. While this level of class imbalance is generally considered acceptable, it might cause the models to slightly bias towards the majority class.

While the amount of data and the class balance is sufficient for training large deep learning models, it is not particularly diverse in terms of the \acp{ISA} present in the dataset. Since all architectures are supported compile targets for recent versions of the Debian Linux distribution, it is likely that these \acp{ISA} are built for running general-purpose operating systems rather than specialized applications such as embedded systems. There are also \ac{ISA} pairs in the dataset that are quite similar, such as `armel`/`armhf` and `powerpc`/`powerpcspe`. This might cause slightly misleading performance numbers when running \ac{LOGO CV}, as the models may overfit to the similarity between these \acp{ISA}.

TODO: most variable width ISAs are from the x86 family

### CpuRec dataset

We use the CpuRec dataset for testing the performance and evaluating the generalizability of our trained models. Contrary to ISAdetect, contains binaries from 76 different \acp{ISA}, providing us with a significantly more diverse set of architectures for evaluating our models.

However, CpuRec only contains a single compiled binary file per \ac{ISA}. We consider this a significant limitation of this dataset. Not only does this mean that the data is too limited to be used for any training of deep learning models, there is also only one sample per \ac{ISA} we can use for evaluating whether the model generalizes to new \acp{ISA}. To claim stronger statistical significance of the per-\ac{ISA} results, we would ideally need more than one sample for each.

Another limitation of the CpuRec dataset is its inconsistencies in labelling, data sourcing, and reproducibility. While the CpuRec repository is open source [@TODO], the origin of the compiled binaries is not clear. For properly labelling this dataset, we have relied on a combination of previous theses and papers using the dataset, inspecting the source code of the CpuRec repository, and using our own tools and processes to determine \ac{ISA} features from the binary code. As a result, we are not fully confident that the labelling of the dataset is completely accurate.

### BuildCross dataset

TODO (Stian)

<!--
5.5.3 BuildCross Dataset

- Library code rather than executables, impact on results
- Which libraries and why, (maybe this should be in methododology?)
- Limited to ELF-supported architectures
- Dependency on external toolchain (mikpe's GitHub)
- Quantity and quality of gathered data
- improves instruction width but not endianness. why?
-->

## Sustainability implications

<!--
- Smaller models use less power which is good
- https://www.ntnu.no/excited/b%C3%A6rekraft-i-it-utdanning
-->

While this thesis focuses on a specialized technical problem in computer science, our contributions may have potential implications for sustainability. We examine some of these implications and relate them to the \ac{SGD} [@UN2015].

Our work contributes to the field of reverse engineering. Reverse engineering is a crucial part of malware analysis and digital forensics. \ac{SGD} 16 targets peace, justice, and strong institutions. Enhanced capabilities in reverse engineering helps combating cybersecurity threats, which impacts this goal in a positive way. However, reverse engineering techniques also have potential for misuse by malicious actors. If reverse engineers with malicious intent discover vulnerabilities in the software, they may use this information to perform illegal activities. While we believe better reverse engineering tools provide a net benefit for software security and thus supports \ac{SGD} 16, there are still considerations to make regarding aiding malicious actors.

Unfortunately, reverse engineering is occasionally used for misusing proprietary software. Malicious actors may steal secrets embedded in compiled code, illegally copy or clone functionality, or bypass licensing mechanisms in the software. As a consequence of this, there is a risk that our work undermines \ac{SGD} 9, which relates to resilient infrastructure and innovation.

The environmental impact of modern AI tools is commonly criticized. Deep learning models require significant amounts of energy, both during training and during inference. TODO finish this

## Limitations

<!--
- Only two target features (time/resource constraint),
  - how that might limit knowledge on how well CNNs in general works on detecting isa features
- Black-box models – hard to interpret why it doesn't generalize that well
- Training on more than just code sections?
- File splitting implications
-->
