# Discussion

## Main findings

### ML ISA detection

Our analysis reveals that machine learning approaches to ISA detection predominantly rely on byte-level features, with two main strategies seeming particularly effective: Byte Frequency Distribution (BFD) and Term Frequency-Inverse Document Frequency (TF-IDF) of N-grams. The most widely adopted approach, BFD introduced by Clemens, uses 256-dimentional feature vector containing counts of the bytes from the input binary was enough to reach an accuracy above 90%. The addition of a simple endianness heuristic increased this to > 98% [@Clemens2015]. While this strategy is simple and cheap in terms of network input size, there were some limitations with this approach also noted by ISAdetect. Base BFD networks struggle separating classes of similar ISAs, such as mips and mipsel [@Kairajarvi2020]. An alternative approach by Sahabandu et al. employs TF-IDF features using combinations of 1-, 2-, and 3-grams. While this method improved upon Clemens' results, it requires a substantially larger feature vector [@Sahabandu2023].

Across the reviewed studies, Support Vector Machines (SVM) and Logistic Regression consistently performs the best. While the main focus on the papers has been feature engineering rather than the specific ML architectures employed, SVM' superior performance points to this classifier being well suited at detecting ISA from byte level n-gram features.

All the papers seem to rely heavily on code section identification during training, usually using binary program header information like ELF. This is highlighted by ELISA and Beckman & Haile, where they dedicate parts of their paper for code section discovery [@Nicolao2018] [@Beckman2020]. ISAdetect attempts to classify full binary programs on classifiers only trained on code-only sections and got comparatively much lower classification scores [@Kairajarvi2020]. We consider this to be a significant finding, as the included literature points to a large difference in classification accuracy when including data-sections in the input binaries.

We have also identified that all the included research attempts to classify ISA's from a list of known ISAs. The other examples of feature detection in ISAs is used to augment multiclass classification of ISAs with is specific endianness heuristics introduced by Clemens and function prologues and epilogues used by ELISA and ISAdetect [@Clemens2015] [@Nicolao2018] [@Kairajarvi2020].

<!--

- Most important feature extraction, all byte level N-grams
- SVM goated, but many works
- ALL on detecting ISA from list of known isas.
- Importance of ELF code section dings, worse performance on whole binaries (ISAdetect)

save for analysis:

Worse performance between some architectures. Addressed by ISAdetect and ELISA, at the cost of less generalizability, featers per included architecture.
In order to combat this, later work like ELISA and ISAdetect experimented with architecture specific features like function prologs and epilogues.

A notable limitation of Sahabandu et al.'s work is the absence of per-architecture classification performance results. Dont know limitations of cross architecture classification of similar architectures.

 -->

### CNN on binary code

The most common application for applying CNN to binary code is malware classification. Thanks to the Malimg dataset and the Microsoft Malware Classification Challenge, researchers have access to high-quality data of sufficient size to train deep learning models. These commonly used datasets also allow for straightforward performance comparisons between the different approaches. We see that a basic encoding of the binary data, where each byte is represented as a pixel value and laid out in a 2D image, combined with a standard CNN architecture, was able to achieve an accuracy of 97% on the Malimg dataset [@Son2022] and 98.7% on Microsoft's [@Yang2018]. More sophisticated techniques such as RGB image encoding [@Alam2024] and ensemble CNN architectures [@Parihar2022] [@Al-Masri2024] pushed the accuracy well past 99.5% for both datasets. 

Moreover, transfer learning approaches have been employed to great success. Comparisons show that a pre-trained VGG-16 architecture with a small fully-connected network performs the same or better than CNNs trained from scratch [@Kumari2017] [@Prima2021]. Another study compared eight different pre-trained CNN architectures and concluded that VGG-16 performed the best [@El-Shafai2021]. 

Two papers used CNN binary analysis for detecting compiler optimization levels. Interestingly, their encoding approach differs from most of the malware classification research. Instead of converting the bytes to an image, they used a one-dimensional CNN with a word embedding layer at the beginning of the network.

## Analysis and interpretation

- Dont need Deep learning for ISA detection
- Architecture specific features, less transferable. Architecture agnostic©.
- Reliance on ELF, is it realistic to always know ELF and .text section, both when training and testing?

Results strongly indicate that CNN architectures can be successfully used for binary analysis tasks. We also note that prior approaches achieve high accuracy even without manual feature engineering. This proves that CNNs indeed are able to automatically discover and learn patterns from binary machine code. However, a limitation in existing literature is that it is predominantly focused on malware. It remains to be seen how transferable these methods are to other binary code classification tasks. In addition, we observe that the malware datasets are quite imbalanced, and some of the existing literature fail to acknowledge this when evaluating the performance. Metrics like class-specific accuracy would be a useful benchmark to reveal whether the models developed exhibit a performance loss with underrepresented classes.

That said, there is reason to believe that methods for detecting of compiler optimization levels might be more transferable to ISA related classification tasks. Since compiler optimization does not alter the behavior of the program, but rather optimize the specific implementation of machine code, we hypothesize that these approaches might be preferred for ISA feature classification.

## Research gaps

We identify multiple research gaps in the current literature. Firstly, while machine learning approaches have been successfully applied to ISA classification applications, these models are trained on a pre-defined set of known architectures. If presented with a software binary of an undocumented ISA, these models would not be able to extract useful information about the architecture. Moreover, the current approaches partially rely on engineering features specific for each instruction set, such as function prologues and epilogues as well as the endian heuristic discussed earlier. This implies that the algorithms developed may not be scalable to other ISAs without further feature engineering work.

Additionally, we are under the impression that deep learning, including CNN, is underutilized for software binary analysis. The strong results from malware classification research provide some evidence that CNNs are able to learn patterns from image representations of software binaries, but this seems under-explored for applications outside the malware space. While some promising attempts at detecting compiler optimization levels using similar techniques have been conducted, the scope of this literature is quite limited.

## Future work

To address the gap in scalability of current ISA detection approaches, we propose researching machine learning models that can identify specific architectural features. Where current methods classifies the full ISA from a set of known architectures, it would be useful to extract properties such as endianness, instruction width, opcode length, register organization, et cetera. This would enable reverse engineers to employ the model for software binaries of unknown architectures.

We hypothesize that deep learning approaches, and CNN in particular, have potential for improving the state of the art in ISA detection of software binaries. By eliminating the need for feature engineering, deep learning algorithms could significantly improve the scalability and scope of the approaches used in current literature.

## Limitations (of our work, methodology)

- Kun scopus
- Bias in papers included from our supervisor.
- IC and QA's affected by researcher bias during screening.
- Qualitiative Research, based on our own experience in the flied, bias shit





<!--

-----------------------------------------------------------------
Notes ml-isa

  - Capturing features (clemens, NLP papers) across multiple bytes when counting require large feature vectors. encoding worked well for n-grams, ie  < 8bit as smallest unit. NLP paper does not have f1 scores, lot of grpahs, but lacks hard numbers for different architectures.
    -
  - SVM performed best, fast learning little data, nice with incomplete binaries reduce viable training data.
 -->
