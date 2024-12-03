# Discussion

## Main findings

The most common application for applying CNN to binary code is malware classification. Thanks to the Malimg dataset and the Microsoft Malware Classification Challenge, researchers have access to high-quality data of sufficient size to train deep learning models. These commonly used datasets also allow for straightforward performance comparisons between the different approaches. We see that a basic encoding of the binary data, where each byte is represented as a pixel value and laid out in a 2D image, combined with a standard CNN architecture, was able to achieve an accuracy of 97% on the Malimg dataset and 98.7% on Microsoft's [@Son2022] [@Yang2018].

- Promising input conversion / code representation:
- Standout architectures
- Standout techniques
  - Transfer learning with VGG-16 seems great

## Analysis and interpretation

- Performance metrics imbalance dataset comparison
- How domain specific is malware classificaton
  - Comment on performance being so high on malimg
  - Dataset imbalance? (might just be true for ml in general)

## Research gaps

<!-- Isa features in general, instruction width, other than endianness etc., unknown isas. -->

## Implications (future work, future use)

Practical and thoretical applications of what we have found

## Limitations (of our work, methodology)

<!--
Notes ml-isa

  - Capturing features (clemens, NLP papers) across multiple bytes when counting require large feature vectors. encoding worked well for n-grams, ie  < 8bit as smallest unit. NLP paper does not have f1 scores, lot of grpahs, but lacks hard numbers for different architectures.
    -
  - SVM performed best, fast learning little data, nice with incomplete binaries reduce viable training data.


 -->
