# Background

<!-- ## Reverse Engineering -->

<!-- TODO -->

## Convolutional Neural Networks

Convolutional neural networks, or CNN, is a machine learning method that builds upon the traditional, fully-connected neural network architectures. Neural networks are built using multiple layers of _perceptrons_, trained using a technique called backpropagation.

CNN provides several significant advantages over fully-connected neural networks:

1. **Parameter efficiency**: Fully-connected neural networks require one model weight per connection between neurons. This means that each added layer in the network typically adds thousands of parameters. In a CNN architecture, each neuron is only connected to a small number of other neurons. Additionally, CNN uses the concepts of _filters_, where the weights in each filter can be shared across the entire input range. Fewer model parameters results in increased memory efficiency and makes the model less prone to overfitting.
2. **Preservation of spatial relationships**: In fully-connected neural networks, there is no mechanism that preserves information about the ordering of the input features. This means that spatial relationships between the input features are not taken into account when training the model. Through the convolution layers and pooling layers, CNN are able to learn spatial features, and even discover them independently of their position in the input.
3. **Automatic feature extraction**: In contrast to most machine learning techniques, CNN does not require manual extraction of features [@cnn-survey]. Eliminating the need for feature engineering can save significantly on time and resources. In addition, this characteristic makes CNN suitable for machine learning applications where humans are not able to discover patterns in the input data themselves.

TODO: Explain the typical layers in a CNN:

- Convolution layer
- Pooling layer
- Dropout
- Convolutional block

TODO: Explain fine-tuning
