# Conclusion

Our systematic review of CNN applications in raw binary code analysis reveals several key findings and trends in the field. The dominance of malware classification applications (18 out of 20 studies) demonstrates both the immediate practical value and the current narrow focus of CNN-based binary analysis. State-of-the-art approaches consistently achieve accuracies above 99% on standard datasets, also when leveraging transfer learning with pre-trained models like VGG-16 or employing ensemble architectures.

We have identified several important trends in our analysis. Image-based binary representations is most common, with recent further innovations in RGB encoding and recurrence plot-based approaches showing promising improvements over traditional grayscale conversions. Transfer learning has proven highly effective, significantly reducing training parameters while maintaining or improving performance, particularly when using established architectures like VGG-16. Specialized CNN architectures, such as MFF-CNN and S-DCNN, demonstrate that domain-specific modifications can enhance performance for binary analysis tasks.

However, our review also identifies several limitations and areas for future research. The heavy reliance on imbalanced datasets like MMCC and Malimg and quality of performance evaluation of some papers raises questions about real-world applicability. Nevertheless, the inclusion of successful application of CNN to compiler optimization detection shows potential for expanding into other binary analysis tasks.

Future research directions should focus on developing more diverse and balanced datasets for malware classification, and exploring applications of CNN to raw binary code beyond malware detection.

The promising results in both malware classification and compiler optimization detection indicate that CNN-based approaches hold significant potential for advancing the field of binary code analysis, particularly in scenarios where traditional reverse engineering techniques are impractical or impossible.
