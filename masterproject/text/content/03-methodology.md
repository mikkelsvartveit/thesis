\acresetall

# Methodology

This chapter describes the methodology used in this thesis. We start by describing the experimental setup, including the system configuration and the datasets used in the thesis in \autoref{experimental-setup}. We then outline the development process for BuildCross, our custom dataset, in \autoref{developing-a-custom-dataset}. Next, we describe the machine learning models, target features, and data preprocessing used in the experiments in \autoref{experiments}. Finally, we present our evaluation strategy and metrics in \autoref{evaluation-strategies}.

## Experimental setup

### Datasets {#methodology-datasets}

This thesis utilizes three primary datasets: BuildCross, ISAdetect, and CpuRec. The BuildCross dataset is a novel contribution of this thesis, and its development discussed in \autoref{developing-a-custom-dataset}. ISAdetect and CpuRec datasets are sourced from previous work in software reverse engineering. These datasets contain samples of binary programs from a variety of different \acp{ISA}. Architectures vary in their similarity regarding features such as endianness, word size, and instruction width, and our model development focuses on the ability to reliably detect architectural features independent of the specific \ac{ISA}. The choice of datasets is therefore motivated by architectural diversity, with the goal of reducing potential correlations between groups of \acp{ISA} and the features we aim to detect. Additionally, since binary programs are not human-readable, errors and inconsistencies in the data are difficult to uncover. We depend on accurate labeling of the datasets to ensure reliable results. Based on our search for appropriate datasets, we have found that the combination of the ISAdetect and CpuRec datasets strikes an optimal balance between the number of architectures represented and the volume of training data available, and they complement each other in a way that aligns with our research objectives.

#### ISAdetect

The ISAdetect dataset is the product of a master's thesis by Sami Kairajärvi and the resulting paper _ISAdetect: Usable Automated Detection of CPU Architecture and Endianness for Executable Binary Files and Object Code_ [@Kairajarvi2020]. One of their key contributions is providing, to our knowledge, the most comprehensive publicly available dataset of binary programs from different \acp{ISA} to date. All program binaries were sourced from Debian Linux repositories, chosen because Debian is a trusted project that has been ported to a wide range of \acp{ISA}. This resulted in a dataset consisting of 23 different architectures. Kairajärvi et al. also focused on addressing the dataset imbalances found in previous research, such as Clemens' work, with each architecture containing approximately 3,000 binary program samples [@Clemens2015]. \autoref{table:isadetect} lists the \acp{ISA} present in ISAdetect and their architectural features.

Table: \acp{ISA} present in ISAdetect dataset \label{table:isadetect}

| ISA        | Endianness | Word size | Instruction width |
| ---------- | ---------- | --------- | ----------------- |
| alpha      | little     | 64        | 32                |
| amd64      | little     | 64        | variable          |
| arm64      | little     | 64        | 32                |
| armel      | little     | 32        | 32                |
| armhf      | little     | 32        | 32                |
| hppa       | big        | 32        | 32                |
| i386       | little     | 32        | variable          |
| ia64       | little     | 64        | variable          |
| m68k       | big        | 32        | variable          |
| mips       | big        | 32        | 32                |
| mips64el   | little     | 64        | 32                |
| mipsel     | little     | 32        | 32                |
| powerpc    | big        | 32        | 32                |
| powerpcspe | big        | 32        | 32                |
| ppc64      | big        | 64        | 32                |
| ppc64el    | little     | 64        | 32                |
| riscv64    | little     | 64        | 32                |
| s390       | big        | 32        | variable          |
| s390x      | big        | 64        | variable          |
| sh4        | little     | 32        | 16                |
| sparc      | big        | 32        | 32                |
| sparc64    | big        | 64        | 32                |
| x32        | little     | 32        | variable          |

The ISAdetect dataset is publicly available through etsin.fairdata.fi [@Kairajarvi_dataset2020]. Our study utilizes the most recent version available at the time of running our experiments, Version 6, released on March 29, 2020. The dataset is distributed as a compressed archive (new_new_dataset/\allowbreak ISAdetect_full_dataset.tar.gz) containing both complete program binaries and code-only sections for each architecture. Additionally, each \ac{ISA} folder contains a JSON file with detailed metadata for each individual binary, including properties such as endianness and word size. Additional labeling used by this thesis was based on another master's thesis by Andreassen and Morrison [@Andreassen_Morrison_2024]. We also expanded the instruction width labeling for undocumented architectures by looking up technical documentation and manuals for the \acp{ISA} in question. Our full set of labels and the differences from prior work are documented in \autoref{table:isadetect-labels-comparison} in the appendix.

#### CpuRec

The CpuRec dataset is a collection of code-only sections extracted from binaries of 72 different \acp{ISA}. It was developed by Louis Granboulan for use with the _cpu_rec_ tool, which employs Markov chains and Kullback-Leibler divergence to classify the \ac{ISA} of input binaries [@Granboulan_paper2020]. Although only one binary per architecture is provided – which is likely insufficient for training a deep learning model on its own – the diversity of \acp{ISA} represented makes this an excellent testing dataset for evaluating our models. \autoref{table:cpurec-labels} lists the \acp{ISA} present in CpuRec and their architectural features.

Table: \acp{ISA} present in CpuRec dataset \label{table:cpurec-labels}

| architecture | endianness | wordsize | instruction width |
| ------------ | ---------- | -------- | ----------------- |
| X86          | little     | 32       | variable          |
| X86-64       | little     | 64       | variable          |
| ARM64        | little     | 64       | 32                |
| Alpha        | little     | 64       | 32                |
| ARMel        | little     | 32       | 32                |
| ARMhf        | little     | 32       | 32                |
| MIPSeb       | big        | 32       | 32                |
| MIPSel       | little     | 32       | 32                |
| PPCeb        | big        | 32       | 32                |
| PPCel        | little     | 64       | 32                |
| HP-PA        | big        | 32       | 32                |
| IA-64        | little     | 64       | variable          |
| M68k         | big        | 32       | variable          |
| RISC-V       | little     | 64       | 32                |
| S-390        | big        | 64       | variable          |
| SuperH       | little     | 32       | 16                |
| SPARC        | big        | unknown  | 32                |
| ARC32el      | little     | 32       | variable          |
| AxisCris     | little     | 32       | 16                |
| Epiphany     | little     | 32       | variable          |
| M88k         | big        | 32       | 32                |
| MMIX         | big        | 64       | 32                |
| PDP-11       | middle     | 16       | variable          |
| Stormy16     | bi         | 32       | variable          |
| V850         | little     | 32       | variable          |
| Xtensa       | bi         | 32       | variable          |
| 6502         | little     | 8        | variable          |
| ARcompact    | little     | 32       | variable          |
| Blackfin     | little     | 32       | variable          |
| FR30         | big        | 32       | 16                |
| i860         | bi         | 32       | 32                |
| MCore        | big        | 32       | 16                |
| MN10300      | little     | 32       | unknown           |
| PIC10        | n/a        | 8        | 12                |
| RL78         | little     | unknown  | unknown           |
| VAX          | little     | 32       | variable          |
| XtensaEB     | big        | 32       | variable          |
| 68HC08       | big        | 8        | variable          |
| Cell-SPU     | bi         | 64       | unknown           |
| FR-V         | unknown    | 32       | 32                |
| Mico32       | big        | 32       | 32                |
| Moxie        | bi         | 32       | variable          |
| PIC16        | n/a        | 8        | 14                |
| ROMP         | big        | 32       | variable          |
| TILEPro      | unknown    | 32       | variable          |
| Visium       | unknown    | unknown  | unknown           |
| Z80          | little     | 8        | variable          |
| 68HC11       | big        | 8        | variable          |
| ARMeb        | big        | 32       | 32                |
| CLIPPER      | little     | 32       | variable          |
| FT32         | unknown    | 32       | unknown           |
| IQ2000       | big        | 32       | unknown           |
| MicroBlaze   | big        | 32       | 32                |
| MSP430       | little     | 16       | variable          |
| PIC18        | n/a        | 8        | 16                |
| RX           | little     | 32       | variable          |
| TLCS-90      | n/a        | 8        | variable          |
| 8051         | n/a        | 8        | variable          |
| CompactRISC  | little     | 16       | variable          |
| H8-300       | n/a        | 8        | variable          |
| M32C         | little     | 32       | variable          |
| MIPS16       | bi         | 16       | 16                |
| NDS32        | little     | 32       | variable          |
| PIC24        | little     | 16       | 24                |
| TMS320C2x    | unknown    | 16/32    | variable          |
| WE32000      | n/a        | 32       | unknown           |
| Cray         | n/a        | 64       | variable          |
| H8S          | unknown    | 16       | variable          |
| M32R         | bi         | 32       | variable          |
| NIOS-II      | little     | 32       | 32                |
| TMS320C6x    | bi         | 32       | 32                |
| ARC32eb      | little     | 32       | variable          |
| AVR          | n/a        | 8        | variable          |
| HP-Focus     | n/a        | 32       | variable          |
| STM8         | n/a        | 8        | variable          |
| TriMedia     | unknown    | 32       | unknown           |

The cpu_rec tool suite is available on GitHub, and the binaries used in this thesis are available in the cpu_rec_corpus directory within the cpu_rec repository [@Granboulan_cpu_rec_dataset2024]. The dataset was curated from multiple sources. A significant portion of the binaries were sourced from Debian distributions, where more common architectures like x86, x86_64, m68k, PowerPC, and SPARC are available. For less common architectures, binaries were collected from the Columbia University Kermit archive, which provided samples for architectures such as M88k, HP-Focus, Cray, VAX, and PDP-11. The remaining samples were obtained through compilation of open-source projects using \ac{GCC} cross-compilers [@Granboulan_paper2020]. Unlike ISAdetect, the CpuRec dataset only labels the name of the \ac{ISA}, without additional architectural features. To address this gap, we referenced Appendix A in the thesis by Andreassen and Morrison, who used this dataset in their work and provided labels for architectural features including endianness, word size, and instruction width specifications for each architecture in the dataset [@Andreassen_Morrison_2024].

However, the documentation of the CpuRec dataset is not as comprehensive as ISAdetect, and the authors of the CpuRec dataset have not provided detailed information about how the binaries were sourced. While we relied on the labeling work by Andreassen and Morrison, we also reviewed technical documentation and manuals available online for all the architectures in question to verify our labeling. Sources used and conclusions drawn in this process are documented in the csv-file used in our source code (/masterproject/code/dataset/cpu_rec-features.csv) [@thesisgithub]. We provide a more detailed discussion on dataset quality in \autoref{dataset-quality-cpurec}. Our labels differ from those of Andreassen and Morrison, and a comparison between them can be found in \autoref{table:cpurec-labels-comparison} in the appendix.

### Technical configuration

For all experiments, we use the Idun cluster at \ac{NTNU}. This \ac{HPC} cluster is equipped with 230 NVIDIA Data Center \acp{GPU} [@Idun]. The following hardware configuration was used for all experiments:

- CPU: Intel Xeon or AMD EPYC (12 cores enabled)
- \ac{GPU}: NVIDIA A100 40GB
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

We find that a batch size of 64 represents a good balance between computational efficiency and model performance. It is large enough to enable efficient \ac{GPU} utilization, while small enough to provide a regularization effect through noise in gradient estimation.

Cross entropy loss is the natural choice for classification tasks, as it tends to provide superior performance for classification tasks compared to mean squared error loss [@Golik2013].

The AdamW optimizer is an improved version of Adam that implements weight decay correctly, decoupling it from the learning rate. It also improves on Adam's generalization performance on image classification datasets [@Loshchilov2019].

A learning rate of 0.0001 is lower than Pytorch's default of 0.001 for AdamW. We make this conservative choice due to early observations showing that small learning rates still cause the AdamW optimizer to reach convergence rather quickly for our dataset. Considering our vast amounts of computational resources, we want to err on the side of slower training rather than risking convergence issues.

A weight decay of 0.01 provides moderate regularization strength, and provides a balance between underfitting and overfitting. It is Pytorch's default for the AdamW optimizer.

## Developing a custom dataset

This thesis introduces BuildCross, a comprehensive toolset and diverse program binary dataset for use by machine learning models in binary analysis. It compiles and extracts code sections from archive files of widely-used open source libraries (referenced in \autoref{table:buildcross-dataset-libraries}). The code sections in the binary files are extracted, in addition to being disassembled for dataset labeling and quality control. We develop BuildCross with the goal of bridging the gap in \ac{ISA} diversity and volume between the ISAdetect and CpuRec datasets. While ISAdetect contains a large volume of binary programs, it consists mostly of architectures from mainstream \acp{ISA}. We believe this dataset alone lacks sufficient diversity to develop truly architecture-agnostic models. CpuRec, on the other hand, contains binaries from a great variety of architectures, but the lack of significant volume and uncertainties with labeling of the dataset makes it unsuited for training larger machine learning models. BuildCross strikes a balance between the two, aiming to generate a larger volume of binary code for the underrepresented, less common architectures.

We have found that large, consistent sources of precompiled binaries for embedded and bare-metal systems are hard to come by, a notion also shared by the authors of ISAdetect and CpuRec [@Kairajarvi2020; @Granboulan_paper2020]. To overcome this limitation and produce a well-documented, correctly labeled dataset, we compile binary programs for uncommon architectures using cross-compilation with \ac{GCC} and GNU Binutils. We develop a pipeline consisting of three steps:

1. creating containerized workable cross-compiler toolchains for different \acp{ISA}
2. gathering compilable source code, configuring the toolchains and compiling binaries
3. extracting features and relevant data from the compiled libraries.

Given ISAdetect's comprehensive architectural coverage, we focus on \acp{ISA} not included in their dataset. Our pipeline is extendable and can incorporate additional target toolchains and binary sources as needed.

### Pipeline for developing toolchains

In order to generate binary programs for specific \acp{ISA}, we need a cross-compiler that can run on our host system and target that architecture. While common targets like x86, ARM and MIPS systems have readily available toolchains for multiple host platforms, the less common architectures not covered by the ISAdetect dataset are, in our experience, either not publicly available or cumbersome to configure properly. We found that the best option was to create and compile these toolchains ourselves, and we decide on \ac{GCC} and GNU Binutils due to the GNU project's long history of supporting a large variety of architectures.

A full cross-compiler toolchain has numerous parts, and since many architectures are unsupported in newer versions of \ac{GCC}, configuring compatible versions of Binutils, LIBC implementations, GMP, MPC, MPFR, etc. would require much trial and error. To get started, we employ the buildcross project by Mikael Pettersson on GitHub, as it contains a setup for building cross-compilers with documented version compatibility for deprecated architectures [@Mikpe2024]. The buildcross project is used as a base for our own toolchain building scripts and expanded to support additional architectures.

The cross-compiler suite uses Apptainer images to create containerized, reproducible, and portable cross-compilation environments for the supported architectures. Apptainer is an open-source alternative to Docker and is designed for high-performance computing environments [@singularity; @singularity_github]. The \ac{GCC} suite's source code with its extensions is around 15GB in size, and to reduce image space and build time, we create a builder image with the necessary dependencies and libraries for building the toolchains. This builder script is used to build the toolchain for each architecture, and the resulting toolchains are stored in separate images of roughly 500MB in size.

### Configuring toolchains and gathering library sources

To streamline the compilation process across multiple target architectures, we use CMake toolchain configuration files rather than manually configuring each library. The manual approach of configuring each library individually for every target architecture is both time consuming and prone to inconsistencies. CMake, a widely used build system, simplifies this process by allowing us to configure and generate build files for different platforms and compilers in a platform-independent way [@cmake]. With CMake, we can specify the target architecture, compiler, linker, and other build options using just one toolchain file per architecture. While most architectures can share a common template toolchain file, CMake also makes it straightforward to implement specific configurations for architectures with unique requirements.

The libraries we select for our dataset are widely used and have large codebases, which provides a good representation of real-world code. They are chosen to ensure that the generated binaries are representative of actual software applications. This is important for training and evaluating our models, as it allows us to assess their performance on realistic data. Additionally, using well-known open source libraries helps us avoid potential issues with licensing, distribution, and reproducibility. By compiling these libraries for the target architectures, we can create a diverse dataset that covers a wide range of instruction sets and architectural features. With the only requirement being that the libraries support CMake, the BuildCross suite can also accommodate additional libraries in the future.

<!-- prettier-ignore -->
| Library   | Version  | Description             |
| :-----     | :---     | ------------------------- |
| freetype [@freetypesource]      | 2.13.3  | A software library for rendering fonts. It is widely used for high-quality text rendering in applications, providing support for TrueType, OpenType, and other font formats [@freetype].                                    |
| libgit2 [@libgit2source]       | 1.9.0   | A portable, pure C implementation of the Git core methods. It provides a fast, linkable library for Git operations that can be used in applications to implement Git functionality without spawning a git process [@libgit2]. |
| libjpeg-turbo [@libjpeg-turbosource] | 3.1.0   | An optimized version of libjpeg that uses SIMD instructions to accelerate JPEG compression and decompression. It is significantly faster than the original libjpeg while maintaining compatibility [@libjpeg-turbo].               |
| libpng [@libpngsource]        | 1.6.47  | The official PNG reference library that provides support for reading, writing, and manipulating PNG (Portable Network Graphics) image files. It is widely used in graphics processing applications [@libpng].               |
| libwebp [@libwebpsource]       | 1.5.0   | A library for encoding and decoding WebP images, Google's image format that provides superior lossless and lossy compression for web images, resulting in smaller file sizes than PNG or JPEG [@libwebp].                     |
| libyaml [@libyamlsource]       | 0.2.5   | A C library for parsing and emitting YAML data. It is commonly used in configuration files and data serialization applications [@libyaml].                                                     |
| pcre2 [@pcre2source]         | 10.45   | Perl Compatible Regular Expressions library (version 2), which provides functions for pattern matching using regular expressions. It is used in many applications for text processing and search operations [@pcre2].     |
| xzutils [@xzutilssource]       | 5.7.1     | A set of compression utilities based on the LZMA algorithm. The XZ format provides high compression ratios and is commonly used for software distribution and archiving [@xzutils].                                           |
| zlib [@zlibsource]          | 1.3     | A software library used for data compression. It provides lossless data-compression functions and is widely used in many software applications for compressing data, including PNG image processing [@zlib].               |

Table: Source libraries used to compile and generate the BuildCross dataset. \label{table:buildcross-dataset-libraries}

The toolchain configuration setup has some flaws, as some of the libraries have dependencies that are not compatible with the target architecture. This is especially true for libraries that are not actively maintained, and the manual labor of patching libraries for each architecture does not scale well for the high number of \acp{ISA}. The most common issue we encounter is the lack of libc intrinsic header file definitions for some of the targets. CMake can in some cases be used to disable some of the library features with missing dependencies, at the cost of in some cases reducing code size. We also compile most architectures with the linker flags `-Wl` and `--unresolved-symbols=ignore-all`, creating binaries that most likely would crash at runtime if the missing symbols were used. Ignoring missing symbols and similar shortcuts still produce valid binaries that are useful for our dataset, as the goal is to create a dataset that is representative of the architectures and their features. Despite this, not all libraries can be compiled for all architectures in time for this thesis, which explains the discrepancies in the amount of data between the architectures.

### Gathering results

The final stage of our pipeline involves extracting and labeling binary data from the compiled libraries. Using CMake's configuring, building, and installing features, we generate install folders containing compiled archive files (.a) for each target architecture. These archive files are collections of compiled binaries (object files) in \ac{ELF} format, providing functions and utilities other programs can link to.

Using the GNU Binutils toolkit from our compiled toolchains, we employ the archiver (ar) to extract individual object files, objcopy to isolate code sections from these objects, and objdump to generate disassembly. This process yield our core dataset of compiled code sections across all target architectures.

For dataset labeling, we extract the endianness and word size metadata directly from each architecture's \ac{ELF} headers. However, determining instruction width proved more challenging due to inconsistent online documentation for uncommon architectures. We establish a method of analyzing instruction patterns in the disassembly, using the hexadecimal mapping between instructions and assembly to infer the size of the instructions. The disassembly output is included in the dataset both for verification of our labeling and as an added utility for the use of BuildCross.

### Final dataset yields and structure

The final dataset spans 40 architectures with approximately 120 MB of binary code, and information on the included architectures can be found in \autoref{table:buildcross-dataset-labels}. The amount of data varies across the architectures, with more supported architectures like arc, loongarch64, and blackfin containing more files, while rarer architectures like xstormy16 and rl78 contain fewer samples due to compilation challenges mentioned in \autoref{configuring-toolchains-and-gathering-library-sources}.

The source code for the cross-compiler suite is available under the masterproject/crosscompiler directory on the thesis GitHub page [@thesisgithub]. The dataset itself is published as a GitHub Release and distributed as a tar.gz file with the following structure:

```{=latex}
\begin{figure}[h]
\begin{minipage}{\textwidth}
\dirtree{%
.1 \textbf{buildcross\_dataset.tar.gz.}.
.2 library\_files/ (Full compiled libraries).
.3 arc/.
.3 bfin/.
.3 (\ldots).
.2 text\_asm/ (Decompiled code sections of libraries).
.3 arc/.
.3 bfin/.
.3 (\ldots).
.2 text\_bin/ (Raw binary of code sections).
.3 arc/.
.3 bfin/.
.3 (\ldots).
.2 labels.csv (Dataset labels for endianness, word size and instruction width).
.2 report.csv (Code section file-sizes for each library in csv format).
.2 report.txt (Code section file-sizes for each library in text format).
}
\end{minipage}
\end{figure}
```

The labels.csv file contains architecture metadata including endianness, word size and instruction width for each binary. The report files provide detailed statistics on code section sizes across libraries, with report.csv offering machine-readable format and report.txt providing human-readable summaries. The data from the labels.csv file is presented in \autoref{table:buildcross-dataset-labels}.

Table: Labels for the \acp{ISA} in the BuildCross dataset, with documented feature values for endianness, word size, instruction width type, and instruction width. Also includes code section sizes extracted for each architecture \label{table:buildcross-dataset-labels}

| architecture | endianness | word size | instruction width type | instruction width | total size (kb) |
| ------------ | ---------- | --------- | ---------------------- | ----------------- | --------------- |
| arc          | little     | 32        | variable               | 16/32             | 3299            |
| arceb        | big        | 32        | variable               | 16/32             | 1729            |
| bfin         | little     | 32        | variable               | 16/32             | 2942            |
| c6x          | big        | 32        | fixed                  | 32                | 5271            |
| cr16         | little     | 32        | variable               | 16/32             | 1583            |
| cris         | little     | 32        | variable               | 16/32             | 4070            |
| csky         | little     | 32        | variable               | 16/32             | 4244            |
| epiphany     | little     | 32        | variable               | 16/32             | 334             |
| fr30         | big        | 32        | variable               | 16/32             | 2215            |
| frv          | big        | 32        | fixed                  | 32                | 5033            |
| ft32         | little     | 32        | fixed                  | 32                | 445             |
| h8300        | big        | 32        | variable               | 16/32             | 4396            |
| iq2000       | big        | 32        | fixed                  | 32                | 2459            |
| kvx          | little     | 64        | variable               | 16/32             | 5012            |
| lm32         | big        | 32        | fixed                  | 32                | 3392            |
| loongarch64  | little     | 64        | fixed                  | 64                | 4814            |
| m32r         | big        | 32        | fixed                  | 32                | 1997            |
| m68k-elf     | big        | 32        | variable               | 16/32/48          | 1866            |
| mcore        | little     | 32        | fixed                  | 16                | 1268            |
| mcoreeb      | big        | 32        | fixed                  | 16                | 1268            |
| microblaze   | big        | 32        | fixed                  | 64                | 5862            |
| microblazeel | little     | 32        | fixed                  | 64                | 5834            |
| mmix         | big        | 64        | fixed                  | 32                | 4305            |
| mn10300      | little     | 32        | variable               | na                | 1251            |
| moxie        | big        | 32        | variable               | 16/32             | 2236            |
| moxieel      | little     | 32        | variable               | 16/32             | 2229            |
| msp430       | little     | 32        | variable               | 16/32             | 223             |
| nds32        | little     | 32        | variable               | 16/32             | 2507            |
| nds32be      | big        | 32        | variable               | 16/32             | 1431            |
| nios2        | little     | 32        | fixed                  | 64                | 4299            |
| or1k         | big        | 32        | fixed                  | 64                | 5541            |
| pru          | little     | 32        | fixed                  | 32                | 2435            |
| rl78         | little     | 32        | variable               | na                | 338             |
| rx           | little     | 32        | variable               | na                | 1486            |
| rxeb         | big        | 32        | variable               | na                | 1300            |
| tilegx       | little     | 64        | fixed                  | 64                | 11964           |
| tilegxbe     | big        | 64        | fixed                  | 64                | 11970           |
| tricore      | little     | 32        | variable               | 16/32             | 1644            |
| v850         | little     | 32        | variable               | 16/32             | 3171            |
| visium       | big        | 32        | fixed                  | 32                | 3481            |
| xstormy16    | little     | 32        | variable               | 16/32             | 219             |
| xtensa       | big        | 32        | variable               | na                | 2671            |

## Experiments

This research primarily involves training, validating, and evaluating \ac{CNN} models using the \ac{ISA} characteristics endianness and instruction width as the target features. This section outlines our approach to data preprocessing as well as the model architectures we use for our experiments.

### Data preprocessing

While most \ac{CNN} architectures are designed for image data, our datasets consist of compiled binary executables. Thus, how these are encoded into a format that can be consumed by a \ac{CNN} is a crucial part of our methodology. In our experiments, we use two different approaches for image encoding.

#### Two-dimensional byte-level encoding

Using two-dimensional byte-level encoding, we treat each byte in the binary file as an integer with values ranging from 0 to 255. These values are arranged in a two-dimensional array of predetermined size and fed into the \ac{CNN}. For files larger than this predetermined size, we use only the initial bytes that fit within the dimensions. Rather than applying data augmentation techniques for files smaller than this predetermined size, we exclude them from the dataset entirely. We discuss this approach in more detail in \autoref{input-size-and-file-splitting}.

When applying two-dimensional \ac{CNN} on 2D grids of this format, the byte values will essentially be treated as pixel values, where the byte sequence forms a grayscale image. \autoref{fig:byte-encoding} shows an example of a 9-byte sequence encoded as a 3x3 pixel grayscale image.

![Encoding bytes as a grayscale image. \label{fig:byte-encoding}](images/methodology/byte-encoding.svg)

This approach was chosen based on previous literature which successfully classified malware from binary executables using \acp{CNN} [@Kumari2017; @Prima2021; @Hammad2022; @Al-Masri2024; @El-Shafai2021; @Alvee2021; @Liang2021; @Son2022].

#### One-dimensional byte-level encoding

Similar to the 2D approach, in one-dimensional byte-level encoding we treat each byte as an integer value, and place them in a one-dimensional array of a predetermined size. If the file is larger than the predetermined size, only the first bytes are used. If the file is smaller than the predetermined size, they are excluded from the dataset.

This approach was chosen based on previous literature which successfully detected compiler optimization levels in binary executables using 1D \acp{CNN} [@Yang2019; @Pizzolotto2021].

#### Input size and file splitting

Since the size of the files in our datasets vary greatly, we need a way to handle varying file sizes. When training our model, we want each data instance to be of a fixed size. Our goal is to use input sizes as small as possible to create time and energy efficient models, while still being large enough to capture enough information about the features we aim to detect. In addition, smaller input sizes ensures that most of our files are large enough to fill the entire input vector, increasing the amount of usable data in the dataset.

For the one-dimensional models, we pick an input size of 512 bytes. We choose this number based on preliminary testing, which revealed that input sizes larger than 512 bytes did not improve model performance.

For the small two-dimensional models, we use an input size of 32x16. This matches the 512-byte input size for the one-dimensional models. In addition, we hypothesize that using a width of 32 bytes might improve the models' ability to detect repeating patterns, since many programs use an instruction width of 32. For the models based on ResNet, we use an input size of 32x32 (which gives 1024 bytes), since this architecture is designed for square images.

For files smaller than the predetermined input size, we choose to exclude them from training rather than padding them to fit the input size. Only the ISAdetect dataset contains files below this threshold, and these represented a small, likely insignificant portion of the total files in that dataset. Furthermore, data augmentation techniques such as padding can introduce noise, and we concluded that exclusion was a better option than potentially degrading the quality of the data. For files in ISAdetect and CpuRec datasets larger than the input size $I$, only the first $I$ bytes from the file is used. For the custom dataset we developed, which contains few but rather large files, we use file splitting to increase the number of training instances. Since these binary files are composed of concatenated code sections from multiple library object files, we consider file splitting to be an appropriate approach. Given a file of size $F$ and a model input size of $I$, each file is divided into $\lfloor F/I \rfloor$ instances.

### Model architectures

In our experiments, we train, evaluate, and compare several model architectures to detect architectural features from binary code. Our approach is inspired by successful applications of \acp{CNN} to binary analysis in previous work, detailed in \autoref{cnn-applications-for-binary-machine-code}. While the scope of this thesis limits the range of model variations we can test, we focus on three factors that have influence model performance in previous research: input encoding, model size and complexity, and embedding layers.

The model architectures described below are specifically designed to investigate how these factors affect a model's ability to learn target \ac{ISA} features. By systematically varying these aspects across our experiments, we aim to determine how effective they are in determining \ac{ISA} features from binary programs.

#### Simple 1D CNN

The smallest model we developed is a small one-dimensional \ac{CNN}. The first layer is a convolution layer of size 1, bringing the filter space dimensionality from 1 to 128 while keeping the spatial dimensions. The rationale for this layer is to align the feature space with the embedding model introduced in \autoref{simple-1d-cnn-with-embedding-layer}. Then, the model consists of three convolutional blocks, each with two convolutional layers and a max pooling layer. After the convolutional blocks comes a global average pooling layer, and a fully-connected block with a single hidden layer for classification. Dropout with a rate of 0.3 is applied after each convolution blocks and between the two fully-connected layers. The full model specification is shown in \autoref{table:simple-1d-cnn}. The model has a total of 152,282 trainable parameters, and is hereby referred to as _Simple1d_.

Table: Simple 1D CNN \label{table:simple-1d-cnn}

| Layer                | Hyperparameters | Output Shape | Parameters |
| -------------------- | --------------- | ------------ | ---------- |
| Input                | –               | (512,)       | –          |
|                      |                 |              |            |
| Convolution 1D       | k=1, s=1, p=0   | (512, 128)   | 256        |
| Dropout              | p=0.3           | (512, 128)   | –          |
|                      |                 |              |            |
| Convolution 1D       | k=3, s=1, p=1   | (512, 32)    | 12,320     |
| Convolution 1D       | k=5, s=2, p=2   | (256, 32)    | 5,152      |
| Max Pooling 1D       | k=2, s=2        | (128, 32)    | –          |
| Dropout              | p=0.3           | (8, 128)     | –          |
|                      |                 |              |            |
| Convolution 1D       | k=3, s=1, p=1   | (128, 64)    | 6,208      |
| Convolution 1D       | k=5, s=2, p=2   | (64, 64)     | 20,544     |
| Max Pooling 1D       | k=2, s=2        | (32, 64)     | –          |
| Dropout              | p=0.3           | (8, 128)     | –          |
|                      |                 |              |            |
| Convolution 1D       | k=3, s=1, p=1   | (32, 128)    | 24,704     |
| Convolution 1D       | k=5, s=2, p=2   | (16, 128)    | 82,048     |
| Max Pooling 1D       | k=2, s=2        | (8, 128)     | –          |
| Dropout              | p=0.3           | (8, 128)     | –          |
|                      |                 |              |            |
| Adaptive Avg Pool 1D | output=1        | (1, 128)     | –          |
| Reshape              | –               | (128,)       | –          |
|                      |                 |              |            |
| Fully Connected      | –               | (8,)         | 1,032      |
| ReLU                 | –               | (8,)         | –          |
| Dropout              | p=0.3           | (8,)         | –          |
| Fully Connected      | –               | (2,)         | 18         |
| Softmax              | –               | (2,)         | –          |

#### Simple 1D CNN with embedding layer

Our one-dimensional word-embedding model builds on the _Simple1d_ \ac{CNN} in \autoref{simple-1d-cnn}, and is constructed by placing an embedding layer at the beginning of the model instead of the size 1 convolution layer. The embedding layer transforms the byte values into a vector of continuous numbers, allowing the model to learn the characteristics of each byte value and represent it mathematically. After the embedding layer, the model is identical to the _Simple1d_ model. The full model specification is shown in \autoref{table:1d-cnn-with-embedding-layer}. This model has a total of 184,794 trainable parameters and is hereby referred to as _Simple1d-E_.

Table: 1D CNN with embedding layer \label{table:1d-cnn-with-embedding-layer}

| Layer                | Hyperparameters | Output Shape | Parameters |
| -------------------- | --------------- | ------------ | ---------- |
| Input                | –               | (512,)       | –          |
|                      |                 |              |            |
| Embedding            | v=256, d=128    | (512, 128)   | 32,768     |
| Dropout              | p=0.3           | (512, 128)   | –          |
|                      |                 |              |            |
| Convolution 1D       | k=3, s=1, p=1   | (512, 32)    | 12,320     |
| Convolution 1D       | k=5, s=2, p=2   | (256, 32)    | 5,152      |
| Max Pooling 1D       | k=2, s=2        | (128, 32)    | –          |
| Dropout              | p=0.3           | (8, 128)     | –          |
|                      |                 |              |            |
| Convolution 1D       | k=3, s=1, p=1   | (128, 64)    | 6,208      |
| Convolution 1D       | k=5, s=2, p=2   | (64, 64)     | 20,544     |
| Max Pooling 1D       | k=2, s=2        | (32, 64)     | –          |
| Dropout              | p=0.3           | (8, 128)     | –          |
|                      |                 |              |            |
| Convolution 1D       | k=3, s=1, p=1   | (32, 128)    | 24,704     |
| Convolution 1D       | k=5, s=2, p=2   | (16, 128)    | 82,048     |
| Max Pooling 1D       | k=2, s=2        | (8, 128)     | –          |
| Dropout              | p=0.3           | (8, 128)     | –          |
|                      |                 |              |            |
| Adaptive Avg Pool 1D | output=1        | (1, 128)     | –          |
| Reshape              | –               | (128,)       | –          |
|                      |                 |              |            |
| Fully Connected      | –               | (8,)         | 1,032      |
| ReLU                 | –               | (8,)         | –          |
| Dropout              | p=0.3           | (8,)         | –          |
| Fully Connected      | –               | (2,)         | 18         |
| Softmax              | –               | (2,)         | –          |

#### Simple 2D CNN

The _Simple2d_ model is a small two-dimensional \ac{CNN}, and aims to test how the two-dimensional encoding of the input effects model performance. The input size is 32x16, which is the result of the 2D encoding of a 512-byte sequence. The first layer is a 1x1 convolution layer, bringing the filter space dimensionality from 1 to 128 while keeping the spatial dimensions. The rationale for this layer is to align the feature space with the embedding model introduced in \autoref{simple-2d-cnn-with-embedding-layer}. Then, the model consists of two convolutional blocks, each with two convolutional layers and a max pooling layer. After the convolutional blocks comes a fully-connected block with a single hidden layer for classification. Dropout with a rate of 0.3 is applied after each convolution blocks and between the two fully-connected layers. The full model specification is shown in \autoref{table:simple-2d-cnn}. The model has a total of 184,794 trainable parameters and is hereby referred to as _Simple2d_.

Table: Simple 2D CNN \label{table:simple-2d-cnn}

| Layer           | Hyperparameters | Output Shape  | Parameters |
| --------------- | --------------- | ------------- | ---------- |
| Input           | –               | (32, 16, 1)   | –          |
|                 |                 |               |            |
| Convolution 2D  | k=1, s=1, p=0   | (32, 16, 128) | 256        |
| Dropout         | p=0.3           | (32, 16, 128) | –          |
|                 |                 |               |            |
| Convolution 2D  | k=3, s=1, p=1   | (32, 16, 32)  | 36,896     |
| Convolution 2D  | k=5, s=2, p=2   | (16, 8, 32)   | 25,632     |
| Max Pooling 2D  | k=2, s=2        | (8, 4, 32)    | –          |
| Dropout         | p=0.3           | (8, 4, 32)    | –          |
|                 |                 |               |            |
| Convolution 2D  | k=3, s=1, p=1   | (8, 4, 64)    | 18,496     |
| Convolution 2D  | k=5, s=2, p=2   | (4, 2, 64)    | 102,464    |
| Max Pooling 2D  | k=2, s=2        | (2, 1, 64)    | –          |
| Dropout         | p=0.3           | (2, 1, 64)    | –          |
|                 |                 |               |            |
| Reshape         | –               | (128,)        | –          |
|                 |                 |               |            |
| Fully Connected | –               | (8,)          | 1,032      |
| ReLU            | –               | (8,)          | –          |
| Dropout         | p=0.3           | (8,)          | –          |
| Fully Connected | –               | (2,)          | 18         |
| Softmax         | –               | (2,)          | –          |

#### Simple 2D CNN with embedding layer

Our two-dimensional embedding model builds on the simple 2D \ac{CNN} model in \autoref{simple-2d-cnn} by placing an embedding layer at the beginning of the model instead of the 1x1 convolution layer. The embedding layer transforms the byte values into a vector of continuous numbers, allowing the model to learn the characteristics of each byte value and represent it mathematically. After the embedding layer, the model is identical to the _Simple2d_ model. The full model specification is shown in \autoref{table:2d-cnn-with-embedding-layer}. This model has a total of 217,306 trainable parameters, and is hereby referred to as _Simple2d-E_.

Table: 2D CNN with embedding layer \label{table:2d-cnn-with-embedding-layer}

| Layer           | Hyperparameters | Output Shape  | Parameters |
| --------------- | --------------- | ------------- | ---------- |
| Input           | –               | (512,)        | –          |
|                 |                 |               |            |
| Embedding       | v=256, d=128    | (512, 128)    | 32,768     |
| Dropout         | p=0.3           | (512, 128)    | –          |
| Reshape         | –               | (32, 16, 128) | –          |
|                 |                 |               |            |
| Convolution 2D  | k=3, s=1, p=1   | (32, 16, 32)  | 36,896     |
| Convolution 2D  | k=5, s=2, p=2   | (16, 8, 32)   | 25,632     |
| Max Pooling 2D  | k=2, s=2        | (8, 4, 32)    | –          |
| Dropout         | p=0.3           | (8, 4, 32)    | –          |
|                 |                 |               |            |
| Convolution 2D  | k=3, s=1, p=1   | (8, 4, 64)    | 18,496     |
| Convolution 2D  | k=5, s=2, p=2   | (4, 2, 64)    | 102,464    |
| Max Pooling 2D  | k=2, s=2        | (2, 1, 64)    | –          |
| Dropout         | p=0.3           | (2, 1, 64)    | –          |
|                 |                 |               |            |
| Reshape         | –               | (128,)        | –          |
|                 |                 |               |            |
| Fully Connected | –               | (8,)          | 1,032      |
| ReLU            | –               | (8,)          | –          |
| Dropout         | p=0.3           | (8,)          | –          |
| Fully Connected | –               | (2,)          | 18         |
| Softmax         | –               | (2,)          | –          |

#### ResNet50

ResNet is a common \ac{CNN} architecture that utilize _residual blocks_, which are groups of convolutional layers with skip connections [@ResNet]. ResNet comes in several variants, and we choose to use the variant with 50 weighted layers, commonly referred to as ResNet50.

The overall architecture of ResNet50 has:

- 1 convolutional layer
- 16 residual blocks, each containing 3 convolutional layers
- 1 average pooling layer
- 1 fully connected layer

To preprocess our data for ResNet50, we use the 2D image encoding described in \autoref{two-dimensional-byte-level-encoding}, with a 32x32 image size. However, since ResNet expects a three-channel (RGB) image, we duplicate the pixel values to all three channels, which essentially results in a grayscale image. The ResNet50 model from the PyTorch Torchvision library is used, and has a total of 23,512,130 parameters.

#### ResNet50 with embedding layer

This model architecture builds on the ResNet50 model described in \autoref{resnet50}, but modifies it to include an initial embedding layer. Specifically, the following modifications are made to the standard ResNet50 model:

- Added an embedding layer with vocabulary size 256 and dimension size 128 as the first layer

- Modified the first convolution layer to accept 128 channels instead of 3

The model takes a vector of length 1024 as input, which is reshaped to 32x32 after the embedding layer. The embedding layer increases the size of ResNet50 by a small factor, resulting in a model with 23,936,898 parameters. This model is hereby referred to as _ResNet50-E_.

### Target features

For every model architecture, we will separately train and evaluate model using these two target features:

- **Endianness** – the ordering of bytes in a multi-byte value.
- **Instruction width type** – whether the length of each instruction is fixed or variable.

We choose these features due to their importance in a reverse engineering process – if the reverse engineer can predictably split up the file into instructions of fixed width, it provides a solid starting point for deciphering instruction-boundaries. Knowing the endianness allows the reverse engineer to properly interpret numerical values such as memory addresses.

Additionally, these features have certain technical properties that make them suitable for deep learning models. First, the features themselves have less ambiguous definitions than other \ac{ISA} characteristics such as word size. Both features have clear classes that are well-suited for classification by models such as \acp{CNN}: endianness of a file is typically either big or little, and instruction width is either fixed or variable. Furthermore, the chosen features exhibit consistent byte patterns across the entire binary, allowing for analysis of small segments of binary code at a time.

## Evaluation strategies

### K-fold cross-validation on ISAdetect dataset

As an initial experiment, we use K-fold cross-validation (as described in \autoref{k-fold-cross-validation}) on the ISAdetect dataset to evaluate the performance of our models. We pick a value of 5 folds, a common choice for K-fold cross-validation that balances between computational cost and robustness. It should be noted that this experiment does not provide an indication of how the model performs on previously unseen \acp{ISA}, since all 23 \acp{ISA} are present in the training data for each fold. However, we still include this experiment for the sake of interpreting the general behavior of the models.

### Leave-one-group-out cross-validation on ISAdetect dataset

The most common way to validate machine learning models is by leaving out a random subset of the data, training the model on the remaining data, and then measuring performance by making predictions on the left-out subset. K-fold cross-validation, as described in \autoref{k-fold-cross-validation-on-isadetect-dataset}, is a more robust variant of the train-test split that repeats this process multiple times with different splits. However, our goal is to develop a \ac{CNN} model that is able to discover features from binary executables of unseen \acp{ISA}.

To validate whether our model generalizes to \acp{ISA} not present in the training data, we use \acf{LOGO CV}, using the \acp{ISA} as the groups (see \autoref{leave-one-group-out-cross-validation} for a description of \ac{LOGO CV}). In other words, we train models for validation using binaries from 22 out of our 23 \acp{ISA} from the ISAdetect dataset, using the single held-out group as the validation set.

Since \ac{LOGO CV} trains a distinct model for each fold (one for each held-out \ac{ISA}), we initialize each model with the same random seed across all 23 folds. This ensures identical starting weights, allowing us to attribute performance differences across folds to the difficulty in classifying each held-out ISA rather than to variations in initial conditions. We can then average performance metrics across folds to obtain a measure of how well that particular model configuration generalizes to unseen \acp{ISA}.

### Testing on other datasets

To conduct further performance evaluation on \acp{ISA} not present in ISAdetect, we use the CpuRec dataset (described in \autoref{cpurec}) as well as BuildCross, the dataset we developed ourselves (described in \autoref{developing-a-custom-dataset}). These evaluation strategies follow a train-test format, where we train the models on designated data from a training dataset, and run inference and test model performance on a testing dataset. Evaluating on additional datasets ensures comprehensive validation of model performance on a more diverse set of \acp{ISA}. Our choice of evaluation strategies are based on these factors:

- The quality of the datasets, in terms of labeling and size
- Focus on inference on \acp{ISA} not present in the testing set
- Limit the number of target features, model variations, and evaluation strategies to ensure proper evaluation of the models within the time and resource constraints of this thesis

\autoref{table:evaluation-strategies} shows the three evaluation strategies in which we use the additional datasets. For ease of reference, these evaluation strategies are hereby named _ISAdetect-CpuRec_, _ISAdetect-BuildCross_ and _Combined-CpuRec_.

Table: Evaluation strategies using multiple datasets \label{table:evaluation-strategies}.

| Reference            | Training dataset       | Evaluation dataset |
| -------------------- | ---------------------- | ------------------ |
| ISAdetect-CpuRec     | ISAdetect              | CpuRec             |
| ISAdetect-BuildCross | ISAdetect              | BuildCross         |
| Combined-CpuRec      | ISAdetect + BuildCross | CpuRec             |

### Cross-seed validation

To account for the stochastic nature of deep neural network training, we validate each architecture by training multiple times with different seeds. The seed impacts factors such as weight initialization and data shuffling. By training using different seeds and averaging the performance metrics, we achieve a more reliable assessment of model performance by mitigating fortunate or unfortunate random initializations. Furthermore, we analyze the stability of our model architecture by examining the deviations in results across different initializations.

For the cross-validation evaluation strategies, we repeat the experiments 10 times with different seeds. For the experiments where we test on CpuRec and BuildCross, we repeat the experiments 20 times. To ensure reproducibility, the seeds and run configurations are documented in the source code repository [@thesisgithub].

### Performance metrics and confidence intervals

To quantify the uncertainty in our model performance metrics, we calculate confidence intervals around the accuracy of our models. The confidence interval is a range of values that is likely to contain the true value of a parameter with a certain level of confidence, and the general statistical procedure is described in \autoref{confidence-intervals-for-binary-classification}. In our case, we apply a 95% confidence interval for the mean accuracy of our models across multiple runs. We also calculate mean accuracy and its standard deviation per \ac{ISA}, as well as confusion matrices to identify systematic misclassifications for the different target features.
