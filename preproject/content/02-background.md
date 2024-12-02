# Background

## Software Reverse Engineering

Software reverse engineering (RE) is the process of analyzing and understanding how a piece of software or software system functions. The term's origins lie in industrial hardware analysis, where Chikofsky & Cross note that examining competitors' designs was important for driving innovation and developing new ideas. While hardware reverse engineering typically also includes replicating existing systems, software RE often aims to gain "sufficient design-level understanding to aid maintenance, strengthen enhancement, or support replacement" [@Chikofsky1990]. Despite the expansion of reverse engineering applications over the years, its core purpose in software is still to recover unavailable information about a system's functionality, requirements, design, and implementation.

### Binary programs and Instruction Set Architecture

A Binary program is a computer program written in machine code consisting of binary instructions. The 1s and 0s of the binary program translate directly to hardware instructions on the computer, and is therefore fed directly into the execution units. Programmers create programs in higher level programming languages and use tools like compilers to translate these programs to a machine-readable binary format. The resulting binary code is specific to the target computer architecture, meaning a program compiled for one processor type cannot directly run on a different architecture.

Instruction Set Architecture (ISA) is a contract between hardware and software on how binary code should be run on the CPU. An important part of this contract is the instruction set, which defines the binary encoding of different instructions, providing a mapping of which bits and bytes translates to which instructions. Each instruction typically has a human-readable key-word (like 'ADD' or 'MOV'), forming an assembly language that allows programmers to understand and write code at the machine level. In addition to defining an instruction set, the ISA gives a complete specification about how software interfaces with hardware, including how instructions can be combined, memory organization and addressing, supported datatypes, memory consistency models etc. Some examples of well known ISA families are x86, ARM and RISC-V, and compilers can typically target specific ISAs allowing the same code to be execution on different architectures.

### Binary Analysis

Binary analysis is software RE applied to binary programs, focusing on analyzing machine code in its executable form. Since a binary program represents software at its lowest level of abstraction, binary analysis is often the first step in reverse engineering such software. Understanding these programs requires knowledge of the target instruction set architecture, as the ISA provide a one-to-one mapping between binary code and computer instructions. These instructions can then be directly translated to a higher-level language that is more easily understood by humans like assembly, enabling further analysis to uncover the functionality and design of the software.

## Convolutional Neural Networks

Convolutional neural networks, or CNN, is a machine learning method that builds upon the traditional, fully-connected neural network architectures. CNN provides several significant advantages over fully-connected neural networks:

1. **Parameter efficiency**: Fully-connected neural networks require one model weight per connection between neurons. This means that each added layer in the network typically adds thousands of parameters. In a CNN architecture, each neuron is only connected to a small number of other neurons. Additionally, CNN uses the concepts of _filters_, where the weights in each filter can be shared across the entire input range. Fewer model parameters results in increased memory efficiency and makes the model less prone to overfitting.
2. **Preservation of spatial relationships**: In fully-connected neural networks, there is no mechanism that preserves information about the ordering of the input features. This means that spatial relationships between the input features are not taken into account when training the model. Through the convolution layers and pooling layers, CNN are able to learn spatial features, and even discover them independently of their position in the input.
3. **Automatic feature extraction**: In contrast to most machine learning techniques, CNN does not require manual extraction of features [@cnn-survey]. Eliminating the need for feature engineering can save significantly on time and resources. In addition, this characteristic makes CNN suitable for machine learning applications where humans are not able to discover patterns in the input data themselves.

TODO: Explain the typical layers in a CNN:

- Convolution layer
- Pooling layer
- Dropout
- Convolutional block

TODO: Explain fine-tuning
