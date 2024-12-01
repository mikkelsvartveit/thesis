# Background


## Software Reverse Engineering

Software reverse engineering (RE) is the process of analyzing and understanding how a piece of software or software system functions. The term's origins lie in industrial hardware analysis, where Chikofsky & Cross note that examining competitors' designs was important for driving innovation and developing new ideas. While hardware reverse engineering typically also includes replicating existing systems, software RE often aims to gain "sufficient design-level understanding to aid maintenance, strengthen enhancement, or support replacement" [@Chikofsky1990]. Despite the expansion of reverse engineering applications over the years, its core purpose in software is still to recover unavailable information about a system's functionality, requirements, design, and implementation.

### Binary programs and Compiled Software 
### Instruction Set Architecture 
### Binary Analysis

Binary analysis is software RE applied to binary programs, and focusing on software only readable by computers. A binary program is a computer program at the lowest level, and binary analysis is the first step in reverse engineering such a piece of software.  At this level, software is read directly by computers, where the 1s and 0s of the program translate directly to specific instructions to execute. Understanding these programs requires knowledge of the target instruction set architecture, which provides a one-to-one mapping between binary code and computer instructions. These can then be directly translated to a higher-level language that is readable by humans like assembly, enabling further analysis to uncover the functionality and design of the software. 


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
