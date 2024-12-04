# Discussion

## Main findings

### Machine learning for ISA detection

Our analysis reveals that machine learning approaches to ISA detection predominantly rely on byte-level features, with two main strategies seeming particularly effective: Byte Frequency Distribution (BFD) and Term Frequency-Inverse Document Frequency (TF-IDF) of N-grams. Clemens showed that BFD alone was enough to achieve an accuracy of above 90%, while the addition of a simple endianness heuristic increased this to above 98% [@Clemens2015]. While this strategy is easy to reason about and cheap in terms of dimensionality, certain limitations were noted by ISAdetect. Base BFD networks struggle separating similar ISAs, such as MIPS and MIPSEL [@Kairajarvi2020]. An alternative approach by Sahabandu et al. employs TF-IDF features using combinations of 1-grams, 2-grams, and 3-grams. While this method improved upon Clemens' results, it requires a substantially larger feature vector [@Sahabandu2023].

Across the reviewed studies, Support Vector Machines (SVM) and Logistic Regression are the machine learning algorithms that consistently perform the best. While the main focus of the studies has been feature engineering rather than the specific machine learning algorithms employed, SVM's superior performance points to this classifier being well suited at detecting ISA from byte-level N-gram features.

All the studies reviewed seem to rely heavily on code section identification during training, usually using binary program header information from the ELF file format. This is highlighted by ELISA and Beckman & Haile, which both dedicate parts of their paper to code section discovery [@Nicolao2018] [@Beckman2020]. ISAdetect attempts to classify full binary programs with classifiers trained on code-only sections, and their evaluation shows significantly lower classification scores in this case [@Kairajarvi2020]. We consider this a significant finding, as the included literature indicates a large difference in classification accuracy when including non-code sections in the input binaries.

We also note that all the included research classifies ISAs from a list of known architectures. The other examples of feature detection in ISAs is used to augment multiclass classification of ISAs with is specific endianness heuristics introduced by Clemens and function prologues and epilogues used by ELISA and ISAdetect [@Clemens2015] [@Nicolao2018] [@Kairajarvi2020]. <!-- TODO: Mikkel skjønte ikke den siste setningen helt -->

<!-- TODO:

- Most important feature extraction, all byte level N-grams
- SVM goated, but many works
- ALL on detecting ISA from list of known isas.
- Importance of ELF code section dings, worse performance on whole binaries (ISAdetect)

save for analysis:

Worse performance between some architectures. Addressed by ISAdetect and ELISA, at the cost of less generalizability, featers per included architecture.
In order to combat this, later work like ELISA and ISAdetect experimented with architecture specific features like function prologs and epilogues.

A notable limitation of Sahabandu et al.'s work is the absence of per-architecture classification performance results. Dont know limitations of cross architecture classification of similar architectures.

 -->

### CNN for binary code analysis

The most common application for applying CNN to binary code is malware classification. Thanks to the Malimg dataset and the Microsoft Malware Classification Challenge, researchers have access to high-quality data of sufficient size to train deep learning models. These commonly used datasets also allow for straightforward performance comparisons between the different approaches. We see that a basic encoding of the binary data, where each byte is represented as a pixel value and laid out in a 2D image, combined with a standard CNN architecture, was able to achieve an accuracy of 97% on the Malimg dataset [@Son2022] and 98.7% on Microsoft's [@Yang2018]. More sophisticated techniques such as RGB image encoding [@Alam2024] and ensemble CNN architectures [@Parihar2022] [@Al-Masri2024] pushed the accuracy well past 99.5% for both datasets.

Moreover, transfer learning approaches have been employed to great success. Comparisons show that a pre-trained VGG-16 architecture with a small fully-connected network performs the same or better than CNNs trained from scratch [@Kumari2017] [@Prima2021]. Another study compared eight different pre-trained CNN architectures and concluded that VGG-16 performed the best [@El-Shafai2021].

Two papers used CNN binary analysis for detecting compiler optimization levels. Interestingly, their encoding approach differs from most of the malware classification research. Instead of converting the bytes to an image, they used a one-dimensional CNN with a word embedding layer at the beginning of the network.

## Analysis and interpretation

<!-- - TODO: Architecture specific features, less transferable. Architecture agnostic©. -->

The reviewed methods for ISA detection face two main limitations, namely their difficulty in separating similar architectures and potential lack of transferability to larger architecture sets. While the feature engineering approaches demonstrated strong performance even with basic linear classifiers, byte level n-gram feature based methods come with apparent challenges. BFD appears to struggle when separating classes of similar architectures, and this is a problem that ELISA and ISAdetect attempts to mitigate using architecture specific features [@Nicolao2018] [@Kairajarvi2020]. BFD works best on architectures with very distinct instruction encodings. We suspect that in case of IoT, embedded devices, and custom hardware, BFD based approaches would struggle to separate classes of ISA extensions tailored for specific devices. Architecture specific feature engineering for each new included class of ISA does not appear to scale well in this context. Sahabandu et al.'s TF-IDF approach has more potential in this case, as similar architectures with subtle differences can be captured by the TF-IDF. However, a major limitation of Sahabandu et al.'s paper is the lack of reported classification performance of each architecture. This makes it difficult to compare how well their method performed on similar architectures [@Sahabandu2023].

Deep learning approaches could potentially eliminate the need for architecture-specific feature engineering through automatic feature discovery. Based on the included ML-ISA papers, there was not an apparent need for deep learning for ISA classification from a list of known ISAs. However, architecture agnostic methods offers significant advantages in terms of transferability and scalability to new architectures. The main drawback with deep learning is the reliance on large datasets and longer training times compared to traditional machine learning approaches. However, the CNN-BCA part of the review highlights the potential in strategies like dataset augmentation and transfer learning, which could improve feasibility of a deep learning approach.

<!-- - Reliance on ELF, is it realistic to always know ELF and .text section, both when training and testing? -->
<!-- Er "lack of transparancy" litt harsh? -->

Code section detection and reliance on header information has shown to impact the byte level n-gram features greatly. ISAdetect highlights this discrepancy when evaluating BFD's ability to work on real world scenarios where header information might be missing [@Kairajarvi2020]. Based on this, it appears that data sections provide detrimental amounts of noise to byte level n-gram features. The studies by Ma et al. and Sahabandu et al. seem to largely ignore this issue when discussing their results, only briefly mentioning that they are training on code sections only [@Ma2019] [@Sahabandu2023]. From our analysis, Beckman and Haile's strategy of using uncertainty in classification across a sliding window seems to be a promising and simple code section detection method. This strategy does require training on code sections only in order for the model to be certain when analyzing sections of code [@Beckman2020]. However, none of the proposed methods found a way around this limitation anyway. ELISA's code section detection method seems promising at the cost of more implementation complexity. However, their approach requires the classification of ISA before code section identification [@Nicolao2018]. In general, we question the availability of code section location information if expanding the set of targeted ISAs. The current approaches demonstrate heavy reliance on header information that may not be consistently available.

<!-- Hva med ISA specific features? kanskje kun research gaps -->

Our results strongly indicate that CNN architectures can be successfully used for binary analysis tasks. We also note that prior approaches achieve high accuracy even without manual feature engineering. This proves that CNNs indeed are able to automatically discover and learn patterns from binary machine code. However, a limitation in existing literature is that it is predominantly focused on malware. It remains to be seen how transferable these methods are to other binary code classification tasks. In addition, we observe that the malware datasets are quite imbalanced, and some of the existing literature fail to acknowledge this when evaluating the performance. Metrics like class-specific accuracy would be a useful benchmark to reveal whether the models developed exhibit a performance loss with underrepresented classes.

That said, there is reason to believe that methods for detecting of compiler optimization levels might be more transferable to ISA related classification tasks. Since compiler optimization does not alter the behavior of the program, but rather optimize the specific implementation of machine code, we hypothesize that these approaches might be preferred for ISA feature classification.

## Research gaps

We identify multiple research gaps in the current literature. Firstly, while machine learning approaches have been successfully applied to ISA classification applications, these models are trained on a pre-defined set of known architectures. If presented with a software binary of an undocumented ISA, these models would not be able to extract useful information about the architecture. Moreover, the current approaches partially rely on engineering features specific for each instruction set, such as function prologues and epilogues as well as the endian heuristic discussed earlier. This implies that the algorithms developed may not be scalable to other ISAs without further feature engineering work. We also note that most of the reviewed studies rely on readable header information from the binaries.

Additionally, we are under the impression that deep learning, including CNN, is underutilized for software binary analysis. The strong results from malware classification research provide some evidence that CNNs are able to learn patterns from image representations of software binaries, but this seems under-explored for applications outside the malware space. While some promising attempts at detecting compiler optimization levels using similar techniques have been conducted, the scope of this literature is quite limited.

## Future work

To address the gap in scalability of current ISA detection approaches, we propose researching machine learning models that can identify specific architectural features. Where current methods classifies the full ISA from a set of known architectures, it would be useful to extract properties such as endianness, instruction width, opcode length, register organization, et cetera. This would enable reverse engineers to employ the model for software binaries of unknown architectures.

We hypothesize that deep learning approaches, and CNN in particular, have potential for improving the state of the art in ISA detection of software binaries. By eliminating the need for feature engineering, deep learning algorithms could significantly improve the scalability and scope of the approaches used in current literature.

## Limitations

<!--
- Kun scopus
- Bias in papers included from our supervisor.
- IC and QA's affected by researcher bias during screening.
- Qualitiative Research, based on our own experience in the flied, bias shit
-->

A potential limitation of our research is using Scopus [@Scopus] as our only database when gathering data for the review. Limiting ourselves to one database had the advantage of a standardized exporting format. We used this to great effect when structuring the primary study selection and data extraction from the included articles. We consider Scopus to be a reputable source of high quality, peer-reviewed papers from most major journals and conferences. However, we do acknowledge that a single database increases the likelihood of bias in the study selection. We consider this risk to be acceptable, since preliminary searches found Scopus to encompass all relevant articles from IEEE, ACM and Science Direct, which also were considered as databases for primary source selection. The ML-ISA part of the review contained two articles not found in our search queries, provided to us by our supervisor. We acknowledge that despite carefully selecting search terms that fit our research questions, there is a risk of having missed relevant research that influence the state of the art.

The application of inclusion criteria and quality assessment on the reviewed research is influenced by us as researchers. While the criteria were phrased such as to remove as much ambiguity as possible to keep our work reproducible, certain inclusion criteria, as well as quality assessment in general, is inherently based on qualitative measures. They are influenced by researcher bias, and our prior experience in the field of software reverse engineering and machine learning might have affected our results. Analysis and interpretation of the results is also mostly based on qualitative measures. While we strive for objectivity and have documented our methodology and approaches, we cannot guarantee that other researchers from the field would reach the exact same conclusions.

<!--
Notes ml-isa

- Capturing features (clemens, NLP papers) across multiple bytes when counting require large feature vectors. encoding worked well for n-grams, ie < 8bit as smallest unit. NLP paper does not have f1 scores, lot of grpahs, but lacks hard numbers for different architectures.
- SVM performed best, fast learning little data, nice with incomplete binaries reduce viable training data.
  -->
