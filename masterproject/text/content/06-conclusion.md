\acresetall

# Conclusion

In this thesis, we have investigated the application of \acp{CNN} for detecting \ac{ISA} features from raw binary code, addressing the challenge of reverse engineering binaries from unknown or undocumented architectures. Our work explores a new direction compared to prior literature, leveraging deep learning models' ability to automatically extract meaningful patterns from the input rather than relying on manual feature engineering.

The goal of RQ1 was to identify which \ac{ISA} features were suitable for detection through \ac{CNN}-based approaches. Our experiments show that for our two tested target features – endianness and instruction width type – the models perform similarly for both features. Using \ac{LOGO CV} on the ISAdetect dataset, we observed accuracies up to 90.3% and 88.0% for endianness and instruction width type classification, respectively. When extending the evaluation with more datasets, we see that accuracy for both target features drops to less than 75% on previously unseen architectures. While we see a significant decline in performance when evaluating on a large set of unseen \acp{ISA}, we do note that the model performance is on par with prior research that relied on carefully engineered features.

RQ2 asked whether the way we encode the binary files for the \ac{CNN} input layer impacts the \acp{CNN}'s ability to learn \ac{ISA} characteristics. Our results revealed that while both one-dimensional and two-dimensional encodings proved viable, neither was consistently better in all scenarios. For endianness detection, the dimensionality of the encoding appeared to have minimal impact, likely because endianness patterns manifest at the byte level regardless of spatial arrangement. Instruction width type detection showed slightly higher accuracies with two-dimensional encodings in some experiments, possibly due to the repeating patterns in some fixed-width instruction sets. However, the performance difference was not large enough to claim a statistically significant advantage for two-dimensional encodings.

For RQ3, for which we experimented with different \ac{CNN} architectures and compared their performance, we conclude that large models with many parameters do not exhibit better performance than very small models for this task. We saw that in nearly all experiments, the small models with less than 150,000 trainable parameters performed on par or better than the large ResNet-50 model with 23.5 million parameters. Additionally, we observed that embedding layers seems to have a positive effect. In many experiments, particularly for endianness detection, the embedding-augmented models showed statistically significant performance improvements over their non-embedding counterparts.

In conclusion, our work has shown that \ac{CNN}-based approaches to detecting individual \ac{ISA} features perform on par with, but not significantly better than, existing approaches in prior research. However, there is value in the automatic feature engineering characteristics of deep learning models over the traditional machine learning approaches due to elimination of significant feature engineering efforts. Through this thesis, we have demonstrated that very small \ac{CNN} work just as well as larger ones for this particular task, as well as proving the effectiveness of embedding layers when applying deep learning techniques to binary code analysis. Lastly, our research has highlighted the importance of high-quality, well-labeled datasets for deep learning applications in binary reverse engineering, and the development of the BuildCross dataset has not only enhanced our own research capabilities but also provides a valuable resource for future work in this field.

## Future work

We identify several possibilities for building on our work. Firstly, extending our approach to more target features would allow for deeper understanding of the binary file. Considering that \ac{CNN}-based models do not demand extensive feature engineering, applying the same methodology for other \ac{ISA} features, such as word size, instruction width (for fixed-width instruction sets), or register count, would be feasible given a dataset with clearly defined labels.

Furthermore, extending our approach to operate on full binary files, rather than just code sections, would be beneficial. This would enable our method to be applied even when the code section of a binary cannot be easily identified. To achieve this, we propose adopting a "rolling window" technique, as demonstrated in a previous binary analysis study by Beckman & Haile [@Beckman2020]. In their approach, a 10 KB segment of the binary is repeatedly classified, with the window offset increased by 5 KB each time. By analyzing certainty metrics across these segments, they were able to automatically detect the code section within the binary.

<!-- CNNs for ISA-classifcation. Hard to not get CNNs to not fit to ISAs. Why not let CNN fit to isas? :o(we tried this for a bit, seemed to work, but not tested) -->

<!-- - ISA classification with buildcross -->

<!-- Other CNN variants? Unet for code section segmentation? :o -->

<!-- Expand buildcross, should be possible with CMAKE. -->

<!-- NAtural language processing techniques -->
