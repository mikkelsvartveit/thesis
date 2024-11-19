# Introduction

Reverse engineering is the process of analyzing and understanding how existing software works by examining its compiled form, with the goal of discovering features and functionality from it. Reverse engineering is frequently used for uncovering security vulnerabilities, quality assurance and verification of programs, and analyzing systems for compatibility or interoperability.

Binary analysis and classification is a crucial step in the reverse engineering pipeline. While existing approaches typically rely on disassembly or decompilation as preprocessing steps, recent advances in deep learning have opened new possibilities for direct binary analysis. Convolutional Neural Networks (CNN) are particularly promising thanks to their ability to automatically discover relevant features, eliminating the need for manual feature engineering.

We present a structured literature review examining the application of CNN to binary code analysis, with an emphasis on approaches that operate directly on raw binaries. We focus on methods that bypass the need for reverse engineering or disassembly, as such preprocessing can be computationally expensive and sometimes unreliable, especially for obfuscated binaries. These analysis techniques could potentially be applied to binaries where the instruction set is unknown or undocumented, where reverse engineering is not feasible.

Through a systematic review of 20 primary studies from the Scopus database, we aim to answer the following research question:

> **RQ:** What approaches for applying CNN to raw binary code analysis have been explored in existing literature? How do the identified CNN approaches compare with respect to:
>
> - the applications and types of features they aim to detect?
> - their binary code representation methods?
> - the network architectures and design choices used?
