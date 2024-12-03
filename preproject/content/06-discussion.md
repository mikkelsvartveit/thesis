# Discussion

## Main findings

### ML ISA detection

We have found that all papers use some form of byte N-gram for feature extraction when detecting ISA from binary programs. From our findings it appears that the differences between architectures is can efficiently be captured by analyzing the distrubution of counts of byte N-grams. One of the two most noteble ways of doing this was first introduced by Clemens, where 256 wide feature vector just containing counts of the bytes (BFD) from the input binary was enough to reach an accuracy above 90%, and a simple endiannes huristic increased this to > 98%. While this strategy is simple and cheap in terms of network input size, there were some limitations with this approach also noted by ISAdetect, that reproduced Clemens results. The BFD feature has a hard time seperating classes of similar ISAs, such as mips and mipsel. In order to combat this, later work like ELISA and ISAdetect experimented wit harchitecture specific features like function prologs and epilogs. The second identified most promising strategy, we consider to be Sahabundu et al.'s paper with TF-IDF features. Using a selection of 1 2 and 3 grams and TF-IDF, Sahabundu et al. was able to improve on Clemens results at the cost of a much larger feature vector. However, an important limitation with Sahabundu et al.'s paper is that they do not list classification performance results on individual architectures.

In terms of ML architectures, SVM and Logistic Regression performs the best. The main focus on the papers seem to be on feature engineering rather than specic ML architectures employed, however the constently perfmant results of SVM across all papers points to this classifier being apt at detecting ISA from byte level N-gram features.

We have also identified that all of the included research attempts to classify ISA's from a list of known ISAs. the only counts of ISAs other than multiclass ISa classification, is specific endiannes heuristics and function prologs and epilogs.

All the papers seem to rely heavely on code section identification during training, usually using binary program header information like ELF.

<!--
- Most important feature extraction, all byte level N-grams
- SVM goated, but many works
- ALL on detecting ISA from list of known isas.
- Importance of ELF code section dings, worse performance on whole binaries (ISAdetect)
 -->

### CNN on binary code

The most common application for applying CNN to binary code is malware classification. Thanks to the Malimg dataset and the Microsoft Malware Classification Challenge, researchers have access to high-quality data of sufficient size to train deep learning models. These commonly used datasets also allow for straightforward performance comparisons between the different approaches. We see that a basic encoding of the binary data, where each byte is represented as a pixel value and laid out in a 2D image, combined with a standard CNN architecture, was able to achieve an accuracy of 97% on the Malimg dataset [@Son2022] and 98.7% on Microsoft's [@Yang2018]. More sophisticated techniques such as RGB image encoding [@Alam2024] and ensemble CNN architectures [@Parihar2022] [@Al-Masri2024] pushed the accuracy well past 99.5% for both datasets. Moreover, transfer learning approaches have been employed to great success. Comparisons show that a pre-trained VGG-16 architecture with a small fully-connected network performs the same or better than CNNs trained from scratch [@Kumari2017] [@Prima2021]. Another study compared eight different pre-trained CNN architectures and concluded that VGG-16 performed the best [@El-Shafai2021]. Two papers used CNN binary analysis for detecting compiler optimization levels. Interestingly, their encoding approach differs from most of the malware classification research. Instead of converting the bytes to an image, they used a one-dimensional CNN with a word embedding layer at the beginning of the network.

## Analysis and interpretation

- Dont need Deep learning for ISA detection
- Reliance on ELF, is it realistic to always know ELF and .text section, both when training and testing?
- Architecture specific features, less transferable.
- Performance metrics imbalance dataset comparison
- How domain specific is malware classificaton
  - Comment on performance being so high on malimg
  - Dataset imbalance? (might just be true for ml in general)
- Compiler optimization detection might be more transferable to ISA detection?
  - Their approach is quite different than most of the malware papers

architecture agnostic.

## Research gaps

<!-- Isa features in general, instruction width, other than endianness etc., unknown isas. -->

All reliance on known isas, what to do when presented with an unknown architecture that does not match entirely.
Feature engineering needing architecture specific features, like function epilogues end such, not scaleble.
No Deep learning has been tried. Can it be used to eliminate feature engineeering.

CNNs underutilized or binary code analysis. Mostly malware. Compiler optimization looks promising.

## Implications (future work, future use)

Practical and theoretical applications of what we have found

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
