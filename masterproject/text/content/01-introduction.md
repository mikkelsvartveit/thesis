\acresetall

# Introduction

Software reverse engineering is the process of analyzing compiled binary programs to understand their functionality, structure, and behavior without access to the source code. Encountering compiled programs where the source code is unknown is very common, particularly for proprietary software where the code is considered intellectual property of the developing company. For these programs, reverse engineering can help third-parties identify security vulnerabilities, detect malware, or assure the quality of programs.

The software reverse engineering process is complex, and numerous techniques, frameworks, and tools exist for simplifying these tasks. A crucial step of most reverse engineering pipelines is code discovery and disassembly of the binary file [@Chen2019]. To understand how a program works, one must first identify where in the binary the code section is, and split up the code section into individual machine instructions. To achieve this, knowledge of the \ac{ISA} is critical. The \ac{ISA} defines the contract between hardware and software, specifying how binary code should be executed on a given processor. Common \acp{ISA} include x86, AMD64, ARM, RISC-V, and MIPS [@Ledin2022]. These instruction sets are well-documented, and disassembling binary programs is straightforward with readily available open-source tools.

However, not all CPUs are built on these common instruction sets. In particular, embedded systems require vastly different specifications than general-purpose computers, and often utilize custom \acp{ISA} that are completely undocumented and unknown to reverse engineers [@Spensky2020]. Additionally, certain programs are purposely hard to reverse engineer thanks to virtualization-obfuscation, a technique where programs are compiled to a randomized instruction set and executed through a corresponding custom virtual machine [@Kinder2012; @Liang2018]. With the rise of \ac{IoT} and custom hardware, malware exploiting these new attack vectors presents a significant challenge for reverse engineers: when the specifications of the \ac{ISA} are unknown, straightforward decompiling the program binary is not feasible [@Costin2018; @OrMeir2019]. Moreover, the methods for \ac{ISA} detection outlined in prior literature relies on closed-set classification, that is, the binary can only be classified if its \ac{ISA} is from a predefined list of architectures [@Preproject]. With programs compiled for embedded systems or custom virtual machines, this is often not the case, and we need other techniques in order to successfully reverse engineer these binaries.

Reverse engineers would benefit from an efficient and reliable way of discovering fundamental architectural properties from binary code when the specific \ac{ISA} is unknown or lacks documentation. Uncovering architectural features such as endianness, word size, and instruction width is fundamental for reverse engineering programs of unknown \acp{ISA} [@Chernov2012].

\acp{CNN} is a class of deep learning models for classifying unstructured, grid-based data. The use of convolution layers allow for deep networks that can capture sophisticated relationships without an excessive amount of computational power. Importantly, the nature of \acp{CNN} allow for automatically discovering significant patterns in the input data, without manually engineering features to train on. While \acp{CNN} are primarily used for processing and analyzing visual data like images, prior research has proven the utility of \acp{CNN} in various binary code analysis tasks, particularly within malware detection and classification [@Preproject]. We hypothesize that similar techniques can be used for detecting the \ac{ISA} or uncovering specific architectural features from compiled binary code.

## Objectives and research questions

This thesis investigates whether \acp{CNN} can be trained to detect individual \ac{ISA} features from raw binary code, enabling the analysis of binaries from previously unseen architectures. Where prior research emphasize \ac{ISA} classification, we shift the focus to feature detection, training models to recognize fundamental architectural properties that can be generalized across \ac{ISA} implementations. We leverage \acp{CNN}' ability to automatically and adaptively learn patterns from the input, eliminating the need for manual feature engineering.

The overarching research question that guides this thesis is:

> **RQ:** To what extent can CNNs effectively identify ISA features from raw binary programs without explicit feature engineering?

We break this down into three sub-questions:

> **RQ1**: Which ISA features can be classified with high accuracy by CNNs?
>
> **RQ2**: How does the choice of method for encoding software binaries impact the CNNs' ability to learn ISA characteristics?
>
> **RQ3**: How does the model architecture impact the CNN's ability to learn ISA characteristics?

## Contributions

The main contribution of our work is a comprehensive evaluation of \acp{CNN} for \ac{ISA} feature detection. We train and evaluate six different \ac{CNN} architectures, comparing the behavior of small and large models, as well as evaluating whether including embedding layers improve the classification performance. We implement comprehensive evaluation strategies, including \ac{LOGO CV} and cross-dataset testing, in order to assess how our models perform on binaries from truly unseen \acp{ISA}. We analyze and compare our results to prior work that rely on feature engineering and traditional machine learning techniques, pointing out trade-offs in terms of accuracy, interpretability, data requirements, and computational resources.

Additionally, we contribute to the field of software reverse engineering by developing the BuildCross dataset and cross-compilation framework. This dataset is developed specifically for thoroughly testing and evaluating our proposed \ac{CNN} models. It contains compiled binary code from 40 different \acp{ISA}, acquired by cross-compiling the source code of 9 widely-used open source libraries. This results in roughly 120 MB of raw binary code. The dataset and associated cross-compilation framework is available on GitHub under an open-source \acf{GPL} [^1].

[^1]: [https://github.com/mikkelsvartveit/thesis/releases](https://github.com/mikkelsvartveit/thesis/releases)

## Thesis structure

The remainder of this thesis is organized as follows.

\autoref{background} provides the theoretical foundation for our work. It includes background knowledge of low-level computer software fundamentals, the reverse engineering process, essential machine learning concepts, and an introduction to \acp{CNN}. We also review related work on \ac{ISA} detection and \ac{CNN}-based binary analysis.

\autoref{methodology} describes our experiments and their setup. We start by introducing the datasets, as well as the technical configuration and hyperparameters, used for training and testing our models. Then, we describe the development process for BuildCross, our custom dataset. Finally, we define our six \ac{CNN} architectures, along with the evaluation strategies for measuring the performance of each of them.

\autoref{results} presents our findings for both endianness and instruction width type classification, revealing model performance on both seen and unseen architectures.

\autoref{discussion} analyzes our results in depth. We summarize the key findings, dissect the behavior of each model, and assess their generalizability to unseen data. Then, we compare our approach to prior research. Moreover, we provide a quality assessment of the used datasets. Finally, we address some limitations of our approach, as well as briefly discussing the sustainability implications of our work.

\autoref{conclusion} concludes our thesis, and summarizes how our findings answer the research questions. We also suggest areas for future research.
