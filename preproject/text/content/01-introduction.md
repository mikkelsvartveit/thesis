# Introduction

## Rationale

Software reverse engineering is the process of analyzing and understanding how existing software works by examining its compiled form, with the goal of discovering features and functionality from it [@Chikofsky1990]. We rely on compiled software all the time, often unaware of its underlying source code. For these programs, reverse engineering is crucial for uncovering security vulnerabilities, verifying and assuring quality of programs, and analyzing systems for compatibility or interoperability.

The first step in a reverse engineering pipeline is usually disassembly and code discovery [@Chen2019]. Most software that runs on general-purpose computers and mobile devices use instruction set architectures (ISA) derived from x86 or ARM [@Gupta2021]. Thanks to standardized architectures, it is relatively straightforward for a reverse engineer to disassemble binaries, identify the code section, and start analyzing the behavior of programs compiled for these platforms. However, the workflow becomes more challenging when faced with non-standard ISAs that is often found in embedded systems and IoT devices. Typically, these instruction sets are proprietary and undocumented [@Chen2024], which significantly complicates the disassembly process. Previous work by Clemens [@Clemens2015], Ma et al. [@Ma2019], and ISAdetect [@Kairajarvi2020] have attempted to bridge this gap by identifying ISA features from binaries of unknown architectures. However, we have not found any previous work systematically comparing the different machine learning techniques employed for ISA identification.

In the past decade, deep learning has caused significant breakthroughs in computer vision, natural language processing, speech recognition, and medical applications [@Dong2021]. Convolutional neural networks (CNN) is a deep learning algorithm commonly used for visual tasks such as image classification. It started gaining traction in 2012, after AlexNet won the annual ImageNet challenge [@AlexNet]. Whereas traditional machine learning techniques typically require significant engineering efforts, CNN often succeed in automatically identifying and learning patterns from the input, as long as there are sufficient amounts of training data and computational resources available.

Our preliminary literature searches indicate that CNN has been applied to software binary input data in previous research, particularly within the domain of malware detection and classification [@Khan2020] [@Son2022] [@Chaganti2022]. However, we identify a research gap in secondary studies related to this topic. As far as we are aware, there are no prior high-quality systematic reviews analyzing primary studies of CNN applied to raw binary code.

## Objectives

This work aims to explore, based on previous research, the viability of leveraging CNN for identifying ISA features from unknown software binaries. We hypothesize that the structure of binary code exhibit spatial patterns that can be discovered and learned by a CNN. With this objective in mind, we present a two-fold systematic literature review. First, we will examine prior attempts at using machine learning techniques for detecting ISA features from compiled software binaries. Then, we will analyze existing research that utilizes CNN for classifying binary code. More specifically, we aim to answer the following research questions:

> **RQ1**: What machine learning approaches have been proposed for discovering ISA information from binary programs? How do the identified approaches compare with respect to:
>
> > **RQ1.1**: the machine learning techniques and architectures employed? \newline
> > **RQ1.2**: their prerequisites, preprocessing, and assumptions about the binary programs? \newline
> > **RQ1.3**: the types of ISA features they can identify? \newline
> > **RQ1.4**: their evaluation and reported performance? \newline

> **RQ2:** What approaches for applying CNN to raw binary code analysis have been explored in existing literature? How do the identified CNN approaches compare with respect to:
>
> > **RQ2.1**: the applications and types of features they aim to detect? \newline
> > **RQ2.2**: their binary code representation methods? \newline
> > **RQ2.3**: the network architectures and design choices used? \newline
> > **RQ2.4**: their evaluation and reported performance? \newline

The main contribution of this project will be a systematic categorization and analysis of prior literature that examines how CNN has been applied to raw binary code, as well as how machine learning has been used for ISA feature detection. We hope to aid researchers in advancing the field of software reverse engineering by systematically presenting previously explored applications, including the state-of-the-art methodologies and their potential limitations.

The rest of this paper is structured as follows: Section [2](#background) provides background information and introduces important concepts. Section [3](#related-work) highlights related work and their contributions. Section [4](#methodology) describes our research methodology, including the search strategy, assessment criteria, and data extraction methods. Section [5](#results) presents the findings from our literature review, categorizing and grouping articles based on applications, methods, and results. Section [6](#discussion) analyzes and interprets the main findings, and provides insights in implications, research gaps, and limitations. Here, we also identify opportunities for future research. Section [7](#conclusion) summarizes and concludes our work.
