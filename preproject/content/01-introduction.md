# Introduction

## Rationale

Binary code analysis is a continually growing field, as our dependance on digital systems continues to increase. We interact with and use compiled binary programs all the time, unaware of where the code ... usecases malware, digital forensics, copyright things etc.With rise of IoT, tons of binares compiled to different instruction set architectures. Current effort of reverse engeneering and analsis require lots of manual work. Usecases: embedded systems, IoT devices, industrial control systems (ICS), automotive systems, mobile devices, cryptographic processors, malware (Introduce more and motivate unknown ISA)

ML great at pattern recognition, Clear use in binary code analysis. Binaries are sequenses of code that relate to eachother, formated spacially next to eac other >>> CNN's

Commissioned by NTNU ... Want to broaden the use of deep neural networks for binary code analysis. To this end we provide a Structured Littereature Review.

## Objectives

In this paper, we have documented and conducted a structured litterature review exploring previous use of convolutional neural networks for binary code analysis. Considering the rise of IoT, our main focus will be on applications of CNN's for binary code analysis without requiring disassembly. With this objective in mind, we want to classify different applications of CNN's to binary code, based on their architecture, encoding of binary code as input, CNN's variations and techniques ... ... ...

The specific objectives of this study is as follows:

- **RQ1**: What machine learning approaches have been proposed for discovering ISA information from binary programs? How do the identified approaches compare with respect to:

  - The machine learning techniques and architectures employed
  - Their prerequisites, preprosseing and assumptions about the binary programs
  - The types of ISA features they can identify
  - The methods performance metrics for discovered ISA features

- **RQ2**: What approaches for applying CNNs to raw binary code analysis have been explored in existing literature? How do the identified CNN approaches compare with respect to:
  - Their binary code representation methods (e.g., how binaries are converted to CNN-suitable input)
  - The network architectures and design choices used
  - The types of features or patterns they aim to detect
  - Their ability to work directly on raw binaries without preprocessing
  - Their generalizability across different analysis tasks

(evaluate research questions a bit)

motivate the structure of the paper

Our main contributions are as follows:
