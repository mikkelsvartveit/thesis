# Introduction

## Rationale

Binary code analysis is a continually growing field, as our dependance on digital systems continues to increase. We interact with and use compiled binary programs all the time, unaware of where the code ... usecases malware, digital forensics, copyright things etc.With rise of IoT, tons of binares compiled to different instruction set architectures. Current effort of reverse engeneering and analsis require lots of manual work. Usecases: embedded systems, IoT devices, industrial control systems (ICS), automotive systems, mobile devices, cryptographic processors, malware (Introduce more and motivate unknown ISA)

ML great at pattern recognition, Clear use in binary code analysis. Binaries are sequenses of code that relate to eachother, formated spacially next to eac other >>> CNN's

Commissioned by NTNU ... Want to broaden the use of deep neural networks for binary code analysis. To this end we provide a Structured Littereature Review.

## Objectives

In this paper, we have documented and conducted a structured litterature review exploring previous use of convolutional neural networks for binary code analysis. Considering the rise of IoT, our main focus will be on applications of CNN's for binary code analysis without requiring disassembly. With this objective in mind, we want to classify different applications of CNN's to binary code, based on their architecture, encoding of binary code as input, CNN's variations and techniques ... ... ...

The specific objectives of this study is as follows:

> **RQ1**: What machine learning approaches have been proposed for discovering ISA information from binary programs? How do the identified approaches compare with respect to:
>
> > **RQ1.1**: the machine learning techniques and architectures employed? \newline
> > **RQ1.2**: their prerequisites, preprocessing, and assumptions about the binary programs? \newline
> > **RQ1.3**: the types of ISA features they can identify? \newline
> > **RQ1.4**: the methods performance metrics for discovered ISA features? \newline

> **RQ2:** What approaches for applying CNN to raw binary code analysis have been explored in existing literature? How do the identified CNN approaches compare with respect to:
>
> > **RQ2.1**: the applications and types of features they aim to detect? \newline
> > **RQ2.2**: their binary code representation methods? \newline
> > **RQ2.3**: the network architectures and design choices used? \newline
> > **RQ2.4**: their evalution and reported performance? \newline

(evaluate research questions a bit)

motivate the structure of the paper

Our main contributions are as follows:
