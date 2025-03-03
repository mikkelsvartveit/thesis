# Methodology

This chapter describes the methodology used in this thesis. We start by describing the experimental setup, including the system configuration and the datasets used in the thesis in \autoref{experimental-setup}. We then describe the machine learning models used in the experiments in \autoref{machine-learning-models}. Finally, we describe our evaluation strategy and metrics in \autoref{evaluation}.

## Experimental setup

### Datasets

This thesis utilizes two primary datasets: ISAdetect and CpuRec, both sourced from previous work in software reverse engineering. These datasets contain samples of binary programs from a variety of different \acp{ISA}. Architectures varies in similarity in terms of features like endianness, word- and instruction size, and our model development focuses on the ability to reliably detect architectural features independent of the specific \ac{ISA}. The choice of datasets is therefore mostly motivated by architectural diversity, in order to reduce the potential correlation between the groups of \acp{ISA} and the features we aim to detect. In addition, binary programs are not human-readable, and errors and inconsistencies in the data are difficult to uncover. We are reliant on accurate labeling in datasets to ensure proper results. From our search, we have found that the combination of datasets ISAdetect and CpuRec strikes a good balance between the number of present architectures and volume of training data. They complement each other in a way that fits our research criteria, and have been validated in previous research.

#### ISADetect

The ISAdetect dataset is the product of a masters thesis by Sami Kairajärvi and the resulting paper: "ISAdetect: Usable Automated Detection of CPU Architecture and Endianness for Executable Binary Files and Object Code" [@Kairajarvi2020]. A part of their contributions is providing, to our knowledge, the most comprehensive publicly available dataset of binary programs from different \acp{ISA} to date. All of their program binaries are collected from Debian Linux repositories, selected due to the Debian distribution being a trusted project and ported to a wide variety of \acp{ISA}. This resulted in a dataset consisting of 23 different architectures. Kairajärvi et al. also focused on tackling the dataset imbalances seen in Clemens' work, and each architecture contains around 3000 binary program samples [@Kairajarvi2020] [@Clemens2015].

The ISAdetect dataset is publicly available through etsin.fairdata.fi [@Kairajarvi_dataset2020]. Our study utilizes the most recent version, Version 6, released March 29. 2020. The dataset is distributed as a compressed archive (new_new_dataset/ISAdetect_full_dataset.tar.gz) containing both complete program binaries and code-only sections for each architecture. Additionally, all of \ac{ISA} folder contains a JSON file with detailed metadata for each individual binary, including properties such as endianness and wordsize. This dataset was used for the same purpouse by Andressaen in his masters thesis, and we referred to his table in Appendix A for additional labeling of instruction width type (fixed/variable) and instruction width ranges [@Andreassen_Morrison_2024].

<!-- TODO: more specific dataset stats based on how we handle the imbalance -->

#### CpuRec

The CpuRec dataset is a collection of executable code-only sections extracted from binaries of 72 different architectures, developed by Louis Granboulan for use with the cpu_rec tool. The cpu_rec uses Markov-chains and Kullback-Leibler divergence with the dataset in order to classify the ISA of an input binary [@Granboulan_paper2020]. Even though only one binary per architecture is provided, which is likely insufficient for training a deep learning model on its own, the diversity of ISAs represented makes the dataset an excellent test set for evaluating our model.

The cpu_rec tool-suite is available on GitHub, and the binaries used in the thesis are available as under cpu_rec_corpus directory [@Granboulan_cpu_rec_dataset2024]. The dataset was curated from multiple sources. A significant portion of the binaries were sourced from Debian distributions, where more common architectures like x86, x86_64, m68k, PowerPC, and SPARC are available. For less common architectures, binaries were collected from the Columbia University Kermit archive, which provided samples for architectures like M88k, HP-Focus, Cray, VAX, and PDP-11. The remaining samples were obtained through compilation open-source projects using a gcc cross-compiler [@Granboulan_paper2020]. Unlike ISAdetect, the CpuRec dataset provides only architecture names without additional feature labels. To fill this gap, we referenced Appendix A of Andreassen's thesis [@Andreassen_Morrison_2024] to obtain architectural features including endianness, wordsize, and instruction width specifications for each architecture in the dataset.

<!-- Keep this info for discussion? Dataset quality
While many of the more common ISAs were packaged using standard file-headers, some of the binaries had undocumented .text sections, where the author had to make educated guesses in order to identify code [source].  -->

### Technical configuration

For all experiments, we use the Idun cluster at \ac{NTNU}. This \ac{HPC} cluster is equipped with 230 NVIDIA Data Center GPUs [@TODO]. The following hardware configuration was used for all experiments:

- CPU: Intel Xeon or AMD EPYC, 12 cores enabled
- GPU: NVIDIA A100 (40 GB or 80 GB VRAM)
- RAM: 16 GB

We use the PyTorch framework for building and training our models. The following software versions were used:

- Python 3.12.3
- PyTorch 2.2.2
- torchvision 0.17.2
- CUDA 12.1
- cuDNN 8.9.2

### Hyperparameters

Unless specified otherwise, we use the training hyperparameters specified in \autoref{table:hyperparameters} for our experiments.

Table: Hyperparameter selection \label{table:hyperparameters}

| Hyperparameter | Value         |
| :------------- | :------------ |
| Batch size     | 64            |
| Loss function  | Cross entropy |
| Optimizer      | AdamW         |
| Learning rate  | 0.0001        |
| Weight decay   | 0.01          |

We find that a batch size of 64 represents a good balance between computational efficiency and model performance. It is large enough to enable efficient GPU utilization, while small enough to provide a regularization effect through noise in gradient estimation.

Cross entropy loss is the natural choice for classification tasks, as it tends to provide superior performance for classification tasks compared to mean squared error loss [@Golik2013].

The AdamW optimizer is an improved version of Adam that implements weight decay correctly, decoupling it from the learning rate. It also improves on Adam's generalization performance on image classification datasets [@Loshchilov2019].

A learning rate of 0.0001 is lower than Pytorch's default of 0.001 for AdamW. We make this conservative choice due to early observations showing that small learning rates still cause the AdamW optimizer to reach convergence rather quickly for our dataset. Considering our vast amounts of computational resources, we want to err on the side of slower training rather than risking convergence issues.

A weight decay of 0.01 provides moderate regularization strength, and provides a balance between underfitting and overfitting. It is Pytorch's default for the AdamW optimizer.

## Machine learning models

This research primarily involves training, validating, and evaluating \ac{CNN} models using ISA characteristics such as endianness, word size, and instruction length as the target features. This subsection outlines our approach to data preprocessing, model architecture selection, and validation techniques.

### Data preprocessing

While most \ac{CNN} architectures are designed for image data, our datasets consist of compiled binary executables. Thus, how these are encoded into a format that can be consumed by a \ac{CNN} is a crucial part of our method. In our experiments, we use two different approaches for image encoding.

#### Two-dimensional byte-level encoding

We treat each byte value as an integer whose value range from 0 to 255. The values are placed in a two-dimensional array of a predetermined size. If the file is larger than the predetermined size, only the first bytes are used. If the file is smaller than the predetermined size, the remaining bytes are padded with zero values.

When applying two-dimensional \ac{CNN} on 2D grids of this format, the byte values will essentially be treated as pixel values, where the byte sequence forms a grayscale image. \autoref{fig:byte-encoding} shows an example of a 9-byte sequence encoded as a 3x3 pixel grayscale image.

![Encoding bytes as a grayscale image. \label{fig:byte-encoding}](images/byte-encoding.svg)

This approach was chosen based on previous literature which successfully classified malware from binary executables using \acp{CNN} [@Kumari2017] [@Prima2021] [@Hammad2022] [@Al-Masri2024] [@El-Shafai2021] [@Alvee2021] [@Liang2021] [@Son2022].

#### One-dimensional byte-level encoding

Similar to the 2D approach, we treat each byte as an integer. The values are placed in a one-dimensional array of a predetermined size. If the file is larger than the predetermined size, only the first bytes are used. If the file is smaller than the predetermined size, the remaining bytes are padded with zero values.

This approach was chosen based on previous literature which successfully detected compiler optimization levels in binary executables using 1D \acp{CNN} [@Yang2019] [@Pizzolotto2021].

### Model architecture

Our research explores various \ac{CNN} architectures to determine the most effective approach for ISA detection. We train and validate several models while experimenting with the following configuration choices.

#### CNN size and complexity

The size of the \ac{CNN} determines the amount of computation required to train the model. While we have significant amounts of computation power available, we need to balance the computational complexity of the model with the available resources as well as the size and diversity of our dataset. We also experiment with both one-dimensional and two-dimensional \ac{CNN} architectures to determine the most effective approach for our dataset.

#### Embedding layers

An embedding layer transforms categorical data into vectors of continuous numbers (see \autoref{embeddings} for details). We attempt treating each byte value as a category, and use an embedding layer at the beginning of the \ac{CNN}. Instead of treating the byte values as numbers, this allows the model to learn the characteristics of each byte value and represent it mathematically.

#### Transfer learning

Transfer learning is a machine learning technique where a model developed for one task is re-used for another task (see \autoref{transfer-learning} for details). Transfer learning is very useful when there is little training data available, as well as in cases of limited computation power or time. Using a transfer learning approach can allow for deep networks despite these constraints. We attempt using \acp{CNN} pre-trained on ImageNet [@ImageNet], and use fine-tuning and feature extraction techniques to create tailored models.

## Evaluation

### Leave-one-group-out cross validation on ISADetect dataset

The most common way to validate machine learning models is by leaving out a random subset of the data, training the model on the remaining data, and then measuring performance by making predictions on the left-out subset. However, our goal is to develop a \ac{CNN} model that is able to discover features from binary executables of unseen \acp{ISA}.

To validate whether our model generalizes to \acp{ISA} not present in the training data, we use \acf{LOGO CV}, using the \acp{ISA} as the groups (see \autoref{leave-one-group-out-cross-validation} for a description of \ac{LOGO CV}). In other words, we train models for validation using binaries from 22 out of our 23 \acp{ISA} from the ISADetect dataset, using the single held-out group as the validation set. Repeating this process for each group and aggregating the results, we get a strong indication of how the model performs on previously unseen \acp{ISA}.

### Testing on other datasets

When a model configuration exhibits high performance under \ac{LOGO CV}, we include it for further performance testing by doing inference on other datasets. Before this stage, a final model is trained using all available training data in ISADetect, that is, without leaving out training instances of any group. Most notably, we will use the CpuRec dataset to test whether our trained model generalizes to unseen \acp{ISA}.

### Cross-seed validation

To account for the stochastic nature of deep neural network training, we validate each architecture by training five times with different random seeds. The seed impacts factors such as weight initialization and data shuffling. By training using different random seeds and averaging the performance metrics, we achieve a more reliable assessment of model performance by mitigating fortunate or unfortunate random initializations. Furthermore, we quantify the stability of our model architecture by examining the standard deviation across different initializations.

### Baseline

- Andreassen
  - Clemens endianness heuristic
