# Introduction

Software reverse engineering is the process of analyzing compiled binary programs to understand their functionality, structure, and behavior without access to the source code. Encountering compiled programs where the source code is unknown is very common, particularly for proprietary software where the code is considered intellectual property of the developing company. For these programs, reverse engineering can help third-parties identify security vulnerabilities, detect malware, or assuring the quality of programs.

The software reverse engineering process is complex, and a myriad of techniques, frameworks, and tools exist for simplifying these tasks. A crucial step of most reverse engineering pipelines is code discovery and disassembly of the binary file. To understand how a program works, one must first identify where in the binary the code section is, and split up the code section into individual machine instructions. Luckily, most general-purpose computers of today use \acp{ISA} based on either x86 or ARM. These instruction sets are well-documented, and disassembling these binary files is straightforward with readily available open-source tools.

## Research questions

The overarching research question that guides this thesis is:

> **RQ:** To what extent can CNNs effectively identify ISA features from raw binary programs without explicit feature engineering?

We break this down into four sub-questions:

> **RQ1**: Which ISA features can be successfully classified by CNNs?

> **RQ2**: How does the model architecture impact the CNN's ability to learn ISA characteristics?

> **RQ3**: How does the choice of binary encoding method impact the CNN's ability to learn ISA characteristics?

> **RQ4**: How does our deep learning approach compare to prior research using other machine learning methods?
