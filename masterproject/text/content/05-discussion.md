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
