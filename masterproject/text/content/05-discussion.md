# Discussion

## Overview of key findings

In this section, we will briefly go over the main findings in our experiments. We will also highlight the potential implications of these findings related to our research questions by indicating further points of discussion for later sections.

<!-- Summarize each experiment, small comment on variability -->

<!--
K-fold
- ALL models perform the same with next to no variance
- The fact that allready seen architectures are included in the test set makes it easy to fit endianness and instruction width to already seen architectures. This backs up our suspicion that the models can easily fit on features that are not inherently tied to endianness and instruction width. We feel it is the correct assumption that LOGO-CV and training and testing on different datasets is the best way of evaluating CNNs general ability to detect ISA features.
-->

In the initial K-fold cross validation experiments on ISAdetect we observed that all models performed similarly well, with next to 100% classification accuracy on both endianness and instruction width type. This result was also achieved with very little variance between runs, and points to the fact that already seen architectures across the training and test set lets the models excel at fitting to endianness and instruction width. The incredibly high accuracy feeds our suspicion that the models can quickly fit to architectural features that are not inherently tied to endianness and instruction width. We believe it was correct to assume that this needed to be investigated further, and that \ac{LOGO CV} and training and testing on different datasets is the best way of evaluating the \acp{CNN} general ability to detect \ac{ISA} features.

In the \ac{LOGO CV} on ISAdetect runs and the evaluation strategies using multiple datasets we see more varied and interesting results compared to K-fold. In terms of raw classification performance, individual model performance varies quite a lot depending on the experimental suite and datasets. In the \ac{LOGO CV} suite with ISAdetect, we are able to achieve <!-- TODO: --> accuracy on endianness detection with and <!-- TODO: --> accuracy on instruction width detection. Both of these scores were achieved with the Simple1d-E model. The performance on the other suites are not quite as convincing, as on the cpu-rec suites we see higher variability between runs, and lower overall accuracy on the BuildCross suite.

<!-- Endiannes, are they able to detect it? which model performs best. 1d Embedding is best-->

<!-- Instruction widht, are they able to detect it? which model performs best 2d is best-->

<!-- Buildcross improves cpu rec performance -->

<!-- ResNet models does not improve prformance -->

<!--
(Stian)
- Superior performance of Simple1d-E
- Bar chart with performance on each ISA
- Statistical significance

<!--
- endianness
- instruction width
- Summary
  - Simple1de endiannes
  - Simple2de instruction width
  - Resnet50 worse than simple
  - Embedding better than no embedding for endainness
  - However, inconclusive results due tu high variability in ... suites.
 -->

### Endianness

<!--
ISAdetect logocv: embedding higher performance across the board
- added complexity of resnet does not improve results
- 1D vs 2D, similar performance, but 1d marginally better.
- Simple 1d & 2d embedding models able to diff between ISAs with same instruction set but different endianness

CPURec: embedding higher performance across the board, but less than ISAdetect logocv
- 1D vs 2D, similar performance, but 2d marginally better.
- Seen archs perform well, with 100% accuracy, but is able to decte archs like blackfin rl78 rx etc very well.
- More correctly guesses than wrong

buildcross: very similar across all models, but 1d no embedding is best
- worse overall performance than cpu rec and isadetect

isadetect & buildcross on cpu rec:
- surprisingly does not improve performance
- 1d embedding is best
- embedding best

Key take aways:
- 1d better at endianness, allthough marginally. Depends on suite, but never performs much worse.
- Embedding seems best for endianness, and performs better on 3/4 cases. Huge diff on LOGO cv isadetect. The roles are reversed when testing on BuildCross though.
- The larger complexity of resnet does not improve results, and usually performs worse.
- Although results seem promesing on LOGO cv for isadetect, the performance on cpu rec and buildcross is not as good.
- Buildcross has no overlap with isadetect, except m68k, and is not as good as in cpu rec.
 -->

<!--TODO: back with statistical significance -->

<!-- Out of all the experiment on the different test suites, Simple1d-E is the model that performed the best overall. While Simple2d-E does marginally beat out Simple1d-E on isadetect-cpurec in \autoref{table:cpurec-endianness-results} and isadetect-buildcross in \autoref{table:buildcross-endianness-results}of 1.2 \ac{p.p.} and 0.5 \ac{p.p.} respectively, we deem these small differences inside **the margin of error**. However, the relatively larger performance wins on LOGO-CV ISAdetect in \autoref{table:logo-endianness-results} of 4.6 \ac{p.p.} is evidence that Simple1d-E would perform best in real world scenarios.

The largest overall performance differences in our experiments is seen when comparing embedding and non-embedding versions of our models. On the endianness experiments embedding models perform better in 3 out of 4 test suites, with a clear gap in performance on logo-isadetect. The only exception ISAdetect-Buildcross where both 1d and 2d non-embedding versions beat out their embedding counterparts, with a lot less variance in the reported average accuracy as well. Still, the large performance difference on LOGO-CV ISAdetect indicates that the embedding models are better suited for endianness detection, where the other results are less significant.

It is clear from our results that the larger complexity of ResNet does not significantly improve results. ResNet50 and ResNet50-E's performance is at least on par with the other models, barring the 95th percentile margin of error. The training loss in the different suites , indicates that the features the ResNet model learns related to endianness are not complex enough to require all the extra complexity and learnable parameters. This is true, even with the larger input of 1024 bytes compared to 512 that the simple models use. -->

### Instruction width

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

![Venn diagram illustrating the overlap of \acp{ISA} present in the ISAdetect, CpuRec and BuildCross datasets \label{fig:dataset-isa-overlap}](./images/discussion/dataset-isa-overlap.png)

This overlap of \acp{ISA} between the datasets is a limitation of our experiments, as it may lead to models memorizing specific \acp{ISA} characteristics rather than learning generalizable features. However, we can mitigate this by excluding the \acp{ISA} present in ISAdetect from the CpuRec dataset, and observe the performance using only the non-overlapping \acp{ISA}. Endianness classification performance after excluding the \acp{ISA} present in ISAdetect from the test set is shown in \autoref{fig:cpurec-endianness-by-model-exclude-overlap}. We observe that the model with the highest accuracy is now Simple2d, achieving an accuracy of 74.7%, down from Simple1d-E's 81.0% when evaluated on the entire CpuRec dataset. For instruction width type classification, the effect of removing overlapping \acp{ISA} is even more pronounced. \autoref{fig:cpurec-instructionwidthtype-by-model-exclude-overlap} shows the instruction width classification performance after excluding the \acp{ISA} present in ISAdetect from the test set. Here, the best-performing model only achieves an accuracy of 44.9%, which is worse than what a dummy model that always predicts the most common class would achieve.

![Endianness classification performance on the CpuRec dataset after excluding the \acp{ISA} present in ISAdetect \label{fig:cpurec-endianness-by-model-exclude-overlap}](./images/discussion/cpurec-endianness-by-model-exclude-overlap.png)

![Instruction width classification performance on the CpuRec dataset after excluding the \acp{ISA} present in ISAdetect \label{fig:cpurec-instructionwidthtype-by-model-exclude-overlap}](./images/discussion/cpurec-instructionwidthtype-by-model-exclude-overlap.png)

We identify several potential reasons for the poor generalizability of our ISAdetect-trained models:

- The diversity of the ISAdetect dataset used for training is quite limited. CpuRec contains 76 different \acp{ISA}, while ISAdetect only contains 23. In addition, the ISAdetect dataset is more homogeneous, with all \acp{ISA} being supported compile targets for recent versions of the Debian Linux distribution. CpuRec, on the other hand, was developed by manually cross-compiling source code to a very diverse set of \acp{ISA}. By inspecting the \ac{ISA} features in the two dataset, we can for instance observe that while all \acp{ISA} in ISAdetect have 32 or 64 bit word sizes, CpuRec also contains several \acp{ISA} with 8 and 16 bit word sizes.

- The CpuRec dataset only contains a single binary file per \ac{ISA}. This is a significant limitation of the dataset that makes our results less conclusive and more sensitive to anomalies in the specific binary used for each \ac{ISA}.

- Due to the nature of deep learning, it is possible that the \ac{CNN} models are picking up on \ac{ISA}-specific patterns that are not inherently related to the endianness or instruction width. This is a common problem in deep learning, and is known as overfitting to the training data. Since it is difficult to interpret the inner workings of \ac{CNN} models, we can only speculate whether this is the case. However, the high accuracies observed when running K-fold cross validation on the ISAdetect dataset do support the claim that the models are easy to fit to full \acp{ISA} compared to fitting them to specific \ac{ISA} characteristics.

As noted in the results chapter, our findings show that augmenting the training data with BuildCross does not improve the generalizability of endianness detection. However, we do see indications that the instruction width type classification task benefits from augmenting the training data with BuildCross. To make this a fair comparison, we must note that training on BuildCross results in more \acp{ISA} overlap between the training and test datasets, as compared to training on ISAdetect only. To emphasize that the performance is actually better on unseen \acp{ISA}, we can examine the results when excluding both the \acp{ISA} present in ISAdetect and BuildCross from the test set. \autoref{fig:combined-instructionwidthtype-by-model-exclude-overlap} illustrates this. Compared to \autoref{fig:cpurec-instructionwidthtype-by-model-exclude-overlap}, we see significant performance improvements across all model architectures, indicating that the inclusion of a more diverse training dataset does improve the generalizability of instruction width type classification. (TODO: reason more about why this is the case)

![Instruction width classification performance on the CpuRec dataset after excluding the \acp{ISA} present in ISAdetect and BuildCross\label{fig:combined-instructionwidthtype-by-model-exclude-overlap}](./images/discussion/combined-instructionwidthtype-by-model-exclude-overlap.png)

#### BuildCross

BuildCross is the dataset we developed specifically for this thesis, containing binaries from 40 different \acp{ISA}. In contrast to the other experiments, we observe that the non-embedding models perform better when evaluated on this dataset. Particularly, the best-performing model for endianness classification is Simple1d, achieving an accuracy of 71.3%. For instruction width type classification, the best-performing model is Simple2d, achieving an accuracy of 69.6%.

An advantage of the BuildCross dataset compared to the CpuRec dataset is that there is little \acp{ISA} overlap with the training dataset (ISAdetect). This reduces the risk of the performance numbers showing up as artificially high due to the models memorizing specific \acp{ISA} characteristics rather than learning generalizable features.

We note that while generalizability for the endianness classification task seem similar between the CpuRec and BuildCross datasets, the instruction width type classification task shows a clear improvement when evaluated on the BuildCross dataset. (TODO reason more about why this is the case.)

## Comparision with prior literature

<!--
(Stian)
- Compare to Andreassen
- Critique of Andreassen
  - Doesn’t exclude previously seen architectures when testing on CPURec
  - Lacking a lot of labels and mislabeling certain things
  -->

## Dataset quality assessment

<!--
5.5.1 ISADetect Dataset

Strengths and limitations
Representation of mainstream vs. exotic architectures
Similar architectures impact logocv and balance? (ref powerpc vs powerpcspe, armel armhf)

5.5.2 CPURec Dataset

Single binary per ISA limitation
Misclassification issues
Statistical reliability concerns

5.5.3 BuildCross Dataset

Library code rather than executables, impact on results
Which libraries and why, (maybe this should be in methododology?)
Limited to ELF-supported architectures
Dependency on external toolchain (mikpe's GitHub)
Quantity and quality of gathered data
improves instruction width but not endianness. why?
-->

## UN sustainability goals

<!--
- Smaller models use less power which is good
- https://www.ntnu.no/excited/b%C3%A6rekraft-i-it-utdanning
-->

## Limitations

<!--
- Only two target features (time/resource constraint),
  - how that might limit knowledge on how well CNNs in general works on detecting isa features
- Black-box models – hard to interpret why it doesn't generalize that well
- Training on more than just code sections?
- File splitting implications
-->
