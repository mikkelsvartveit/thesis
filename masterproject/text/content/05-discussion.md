# Discussion

## Overview of key findings

<!--
(Stian)
- Superior performance of Simple1d-E
- Bar chart with performance on each ISA
- Statistical significance
-->

## Model architecture performance analysis

<!--
(Mikkel)
- Why embeddings work so well
- Why larger models do not perform better
- 1D vs 2D
  - 2D better at instruction width? Why?
- Variability ("flakyness") in model performance
-->

### Impact of embedding layers

In most of our experiments, we see that the model architectures that employ an embedding layer as the first layer of the model perform significantly better than their non-embedding counterparts. This is a key finding, and aligns with our hypothesis that embedding techniques may improve performance for \ac{CNN} models due to the categorical nature of binary code.

Consider this simple instruction for the Intel 8080 instruction set:

```assembly
ADI 25;
```

It uses the `ADI` opcode, which indicates an addition with an immediate value. It sums the content of the accumulator register and the immediate value, and saves the result to the accumulator register. We can examine what this looks like when assembled to a 16-bit binary instruction:
$$
\underbrace{1100\ 0110}_{Opcode} \ \ \underbrace{0001\ 1001}_{Immediate\ value}
$$
The first byte contains the operation code. While operation codes are represented as numbers in the executable code, there is no semantic meaning to this number. It is actually  a discrete, categorical piece of data that have no semantic relationship to bytes of close values such as $1100\ 0101$ and $1100\ 0111$.

Intuitively, an operation that is semantically similar to `ADI` (Add Immediate) is `SUI` (Sub Immediate). It performs the same operation, but subtracts the immediate value from the accumulator instead of adding it. The opcode for `SUI` is $1101\ 0110$. Converting this to base 10, the numbers used to represent the `ADI` and `SUI` instructions are 198 and 214. These values themselves do not properly represent the close semantic relationship between the operations.

However, introducing an embedding layer in the model makes it capable of identifying and learning semantic relationships such as this by converting each byte value into a continuous vector. Bytes with close semantic relationships would be represented as similar vectors. While this is a very simple example, converting categorical data into semantic-capturing vectors is a powerful technique that often results in superior performance when training and testing deep learning models on categorical input.

## Model generalizability

<!--
- LOGO already tests this
  - Show comparison to regular leave-one-out cross validation to prove that it overfits if not using LOGO CV
- Why does this not convert well to CpuRec and BuildCross?
  - Limited sample size
  - Greater diversity of architectures
    - Struggles with 8-bit?
  - Statistical significance
- Does training on BuildCross improve performance?
- Visualize some grayscale images
-->

## Comparision with prior literature

<!--
(Stian)
- Compare to Andreassen
- Critique of Andreassen
  - Doesn’t exclude previously seen architectures when testing on CPURec
  - Lacking a lot of labels and mislabeling certain things
  -->

## Dataset quality assessment

<!--
5.5.1 ISADetect Dataset

Strengths and limitations
Representation of mainstream vs. exotic architectures

5.5.2 CPURec Dataset

Single binary per ISA limitation
Misclassification issues
Statistical reliability concerns

5.5.3 BuildCross Dataset

Library code rather than executables, impact on results
Limited to ELF-supported architectures
Dependency on external toolchain (mikpe's GitHub)
Quantity and quality of gathered data
-->

## UN sustainability goals

<!--
- Smaller models use less power which is good
- https://www.ntnu.no/excited/b%C3%A6rekraft-i-it-utdanning
-->

## Limitations

<!--
- Only two target features
- Black-box models – hard to interpret why it doesn't generalize that well
- Training on more than just code sections?
- File splitting implications
-->
