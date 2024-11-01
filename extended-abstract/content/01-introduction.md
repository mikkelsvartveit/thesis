# Introduction

The analysis of binary code has become critical in reverse engineering and cybersecurity domains. As the diversity of binary programs continues to expand across embedded systems, IoT devices, and various digital platforms, traditional analysis methods face scalability challenges and often lack transferability across domains. While existing approaches typically rely on disassembly or decompilation as preprocessing steps, recent advances in deep learning, particularly Convolutional Neural Networks (CNN), have opened new possibilities for direct binary analysis.

We present a structured literature review examining the application of CNNs to binary code analysis, with an emphasis on approaches that operate directly on raw binaries. We focus on methods that bypass the need for reverse engineering or disassembly, as such preprocessing can be computationally expensive and sometimes unreliable, especially for obfuscated binaries. These analysis techniques could potentially be applied to binaries where the instruction set is unknown or undocumented, where reverse engineering is not feasible.

Through a systematic review of 20 primary studies from the Scopus database, we examine various approaches based on their binary representation techniques, CNN architectures, and target applications.
