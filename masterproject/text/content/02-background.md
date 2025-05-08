# Background

This chapter presents the theoretical foundation required to understand the rest of the thesis. Starting with \autoref{computer-software}, we look at the basic, low-level concepts of computer software, instruction sets, and compilers. \autoref{software-reverse-engineering} then introduces software reverse engineering, which is the overarching topic of this thesis. Finally, \autoref{machine-learning} introduces the necessary concepts from the field of machine learning, with a focus on \acp{CNN}.

## Computer software

### Binary executables

All computer software boils down to a series of bytes readable by the CPU. The bytes are organized in _instructions_. An instruction always includes an _opcode_ (Operation Code), which tells the CPU what operation should be executed. Depending on the opcode, the instruction often contains one or more _operands_, which provides the CPU with the data that should be operated on. The operands can be immediate values (values specified directly in the instruction), registers (a small, very fast memory located physically on the CPU), or memory addresses. \autoref{fig:arm-instruction} illustrates the instruction format of ARM, which uses 32-bit instructions.

![Instruction format and examples from the ARM instruction set. \label{fig:arm-instruction}](images/background/arm-instruction.svg)

<!-- Assembly? -->

### Instruction set architectures

An \ac{ISA} is a contract between hardware and software on how binary code should be run on a given computer. In the early days, every new computer system was created with a new \ac{ISA}, meaning programs had to be custom-written for each specific machine. IBM and their System/360 series, introduced in 1964, were the first to use the \ac{ISA} as an abstraction layer between hardware and software. This new approach meant that despite having different internal architectures, all System/360 computers could run the same programs as they shared a common \ac{ISA}. The commercial success of this approach set an industry standard that continues to define modern computing, where hardware manufacturers can implement already established \acp{ISA}, ensuring cross generational program compatibility.

In addition to defining an instruction set, the \ac{ISA} gives a complete specification about how software interfaces with hardware, including how instructions can be combined, memory organization and addressing, supported data types, memory consistency models, and interrupt handling. Examples of well‐known \ac{ISA} families are x86, ARM, and RISC-V. Compilers can typically target multiple \acp{ISA}, allowing the same high‐level source code to be executed on different architectures through appropriate translation to the target instruction set.

#### CISC and RISC

\acp{ISA} today generally fall into two camps: \ac{CISC} and \ac{RISC}. \ac{CISC} architectures, like x86, provide many specialized instructions that can perform complex operations in a single instruction. \ac{CISC} can simplify complex operations at the programming level as well as potentially reduce code size, but at the cost of requiring more complex hardware. \ac{RISC} architectures, like ARM and RISC-V, favor simplicity with a smaller set of fixed-length instructions that execute in a single cycle, making them potentially more energy-efficient and easier to implement.

#### Instruction set

An important part of all \acp{ISA} is the instruction set, which defines the binary encoding of different instructions, providing a mapping of which bits and bytes translates to which instructions. Each instruction typically has a human‐readable keyword (like 'ADD' or 'MOV'), forming an assembly language that allows programmers to understand and write code at the machine level.

#### Word size

<!-- TODO: Elaborate how it is not clearly defined, could be reffering to register width, addressable memory space size, datapath width in the cpu, ALU input width. In earlier architectures, likely used in marketing aswell -->

A fundamental characteristic of any \ac{ISA} is its word size, which defines the natural unit of data the processor works with – typically 32 or 64 bits in modern architectures. This affects everything from register sizes to memory addressing capabilities.

#### Endianness {#background-endianness}

<!-- TODO: explain different types of endianness seen in diff archs -->

The endianness determines how multi-byte values are stored in memory: little-endian architectures store the least significant byte first (like x86), while big-endian stores the most significant byte first, as illustrated in \autoref{tab:endianness}.

```{=latex}
\begin{table}[h]

\vspace{0.2cm}

\begin{center}

\strong{(a) Big endian}
\vspace{0.1cm}

\begin{tabular}{|c|c|c|c|c|}
\hline
Address & 0x1000 & 0x1001 & 0x1002 & 0x1003 \\
\hline
Byte & 0x12 & 0x34 & 0x56 & 0x78 \\
\hline
\end{tabular}

\vspace{0.4cm}

\strong{(b) Little endian}
\vspace{0.1cm}

\begin{tabular}{|c|c|c|c|c|}
\hline
Address & 0x1000 & 0x1001 & 0x1002 & 0x1003 \\
\hline
Byte & 0x78 & 0x56 & 0x34 & 0x12 \\
\hline
\end{tabular}

\end{center}

\caption{Comparison of how a 32-bit integer is stored in big endian and little endian.}
\label{tab:endianness}

\end{table}
```

#### Instruction width

The instruction width refers to the size, typically measured in bits, of a single CPU instruction. Some architectures, such as ARM64, have _fixed-width instructions_. This means that each instruction has the same size. Others, such as most \ac{CISC} instruction sets, have _variable-width instructions_, where the size of each instruction can vary based on factors such as the opcode and the addressing mode. For instance, x86-64 programs can contain instructions ranging from 8 to 120 bits.

### Compilers

Software developers employ tools like compilers and interpreters to convert programs from human-readable programming languages to executable machine code. In the very early days of computer programming, software had to be written in assembly languages that mapped instructions directly to binary code for execution. Growing hardware capabilities allowed for more complex applications, however, the lack of human readability of assembly languages made software increasingly difficult and expensive to maintain. In order to overcome this challenge, compilers were created to translate human-readable higher-level languages into executable programs. In the early 1950s, there were successful attempts at translating symbolically heavy mathematical language to machine code. The language FORTRAN, developed at IBM in 1957, is generally considered the first complete compiled language, being able to achieve efficiency near that of hand-coded applications. While languages like FORTRAN were primarily used for scientific computing needs, the growing complexity of software applications drove the development of more advanced operating systems and compilers. One such advancement was the creation of the C programming language and its compiler in the early 1970s. Modern compilers (like the C compiler) are able to analyze the semantic meaning of the program, usually through some form of intermediate representation. The \ac{ISA} of the target system provides the compiler with the recipe to translate the intermediate representation into executable code. The intermediate representation is usually language- and system architecture-agnostic, which has the added benefit of allowing a compiler to translate the same program to many computer architectures.

The evolution of compilers brought significant advantages in code portability and development efficiency. Programming languages' increasing abstraction away from machine code was necessary to achieve efficient development and portability across different computer architectures. However, this combined with other transformations done by compilers increasingly widened the gap between the original source code and the binary executable. By separating the program's logic from its hardware-specific implementation, developers could write code once, compile, and run it on every platform they wanted, at the cost of making it more difficult to understand what a binary program does.

### Embedded targets and cross-compilation

Embedded systems are specialized computing devices integrated within larger systems to perform specific and dedicated functions. Unlike general-purpose computers, embedded systems designed for specific tasks are therefore typically optimized for reliability, power efficiency, and cost-effectiveness. These embedded systems power everything from household appliances like refrigerators and washing machines to networking equipment like routers, and a vast array of \ac{IoT} devices. Most embedded systems are characterized by resource constraints, including limited memory, processing power, and energy capacity. In order for these systems to perform in such environments, embedded platforms typically incorporate specialized hardware with custom processors and peripherals optimized for specific tasks. As a result, many embedded systems feature limited or no user interface, operating as headless systems controlled programmatically.

Since these specialized systems frequently use custom hardware with \acp{ISA} different from standard desktop or server computers, they present unique challenges for software development. Unlike general-purpose computers, which are often used to create programs for the same platform it is built on, it's often impractical or impossible to compile code directly on the target device. Developers typically have to use a technique called cross-compilation to build software for embedded systems.

#### Cross-compilation

Cross-compilation is the process of generating executable code for a platform (target) different from the one running the compiler (host). This approach allows developers to use more powerful development systems to create software for resource-constrained target devices.

In cross-compilation terminology, three distinct systems are involved in the process. The host system is the computer system where compilation of programs occurs, providing the computational resources needed for the build process. The target system is where the compiled code will eventually run, often with limited resources that would make direct compilation impractical. The build system refers to the system where the compiler itself was built, which is frequently the same as the host system in most development scenarios.

A _cross-compiler toolchain_ is the collection of software tools necessary to build executables for a target system. A complete toolchain consists of several key components working together. The compiler, such as GCC or Clang, serves as the core tool that converts source code to machine code appropriate for the target architecture. Binary utilities (like Binutils) provide essential tools for creating and managing binary files across different architectures. The C/C++ standard library supplies standard functions and data structures optimized for the target system, while a debugger helps identify and fix issues in the compiled program, often supporting remote debugging capabilities for target hardware.

#### GNU Compiler Collection and GNU Binutils

The \ac{GCC} is a comprehensive compiler system supporting various programming languages including C, C++, and Fortran. \ac{GCC} started out as the GNU C Compiler in 1987, but was later renamed as it expanded to support more languages. \ac{GCC} is designed to be highly portable, and can be built to run on various operating systems and hardware architectures. It features a modular design, using internal intermediate representations that are largely host and target system agnostic. This lets much of the optimization logic and transformation to work with different frontends for different programming languages and different backends to generate code for a wide range of \acp{ISA}. \ac{GCC} by itself takes in an input file in supported languages like C and outputs assembly for the target architecture. Each instance of a \ac{GCC} compiler is configured to target a specific architecture, and this flexibility allows developers compile versions of \ac{GCC} to build software for different platforms.

\ac{GCC} is not able to create working executables by itself however, as behind the scenes \ac{GCC} sets up a pipeline consisting of different tools in order to create executable programs. A common pairing to set up this pipeline with the core \ac{GCC} is GNU Binutils, which is a collection of binary program tools that is designed to work alongside compilers to create and manage executables. Some key components of Binutils include:

- **as**: The GNU assembler, which converts assembly language to machine code
- **ld**: The GNU linker, which combines machine code files into executables or libraries
- **ar**: Creates, modifies, and extracts from archive files (static libraries)
- **objcopy**: Copies and translates object files between formats
- **objdump**: Displays information about object files, including disassembly
- **readelf**: Displays information about ELF format files

When invoking \ac{GCC} to create a final executable, the compiler automatically calls the appropriate Binutils tools to create the final executable, and an illustration of this pipeline can be seen in \autoref{fig:gcc-binutils-pipeline}. Since the final executable depends on the target system, the \ac{GCC} compiler and Binutils tools must be configured to target the same architecture. This is typically done by specifying the target architecture when building the toolchain, after which the \ac{GCC} compiler will use the appropriate Binutils tools to generate the final executable for that architecture.

![Illustration of how the \ac{GCC} with Binutils pipeline when compiling a source program with gcc. \label{fig:gcc-binutils-pipeline}](images/background/gnu-gcc-binutils-pipeline.svg)

All of the GNU projects are distributed under the \ac{GPL}, which allows users to freely use, modify, and distribute the software. This has made \ac{GCC} and Binutils widely adopted in open-source projects and embedded systems development.

#### Binary file formats and structures

Binary files come in different formats, each with its own structure and purpose, and understanding these formats is necessary for working with compiled code. In order for a program to be interpreted or executed by a computer it must be stored in a way that the CPU can read and understand. One way of doing this in to include a header at the beginning of the file, that contains metadata about the executable. While different file formats exists, like Portable Executable for Windows and Mach Object (Mach-O) for macOS, the most common binary file format for Unix-like systems and many embedded devices is the \ac{ELF}. Since \ac{ELF} format is the most commonly seen in cross-compilation and embedded systems, it is the focus of this section.

The \ac{ELF} format is a flexible and extensible binary file format that can be used for executables, object code, shared libraries et cetera. The \ac{ELF} header contains information about the file type, \ac{ISA}, entry point address for where to start execution, and section headers. It also includes support for debugging information, symbol tables, and relocation entries, making it easier to analyze and debug programs.

\ac{ELF} files are organized into what is called sections, which are contiguous blocks of information within the binary file serving different purposes. The section names are conventionally prefixed with a period (.), and common sections in \ac{ELF} files include:

- **.text**: Contains the executable code (machine instructions). Sometimes referred to as the code section.
- **.data**: Contains initialized global variables
- **.bss**: Contains uninitialized global variables
- **.rodata**: Contains read-only data like strings
- **.symtab**: Symbol table with function and variable names and their locations

File formats like \ac{ELF} provides a standardized way to represent the structure of a binary file, including sections for code, data, and metadata. When cross-compiling using the \ac{GCC}/Binutils suite, \ac{ELF} files are typically the default file format when targeting embedded targets. Therefore, \ac{ELF} headers are typically found in binary files that serve different purposes in the creation of an executable program:

- **Object files**: Compiled source code files that contain machine code but are not yet executable. They must be linked together to create complete programs, as illustrated in \autoref{fig:gcc-binutils-pipeline}.

- **Executable files**: The final output of the compilation process that can be directly run by the operating system. These files contain the complete program with all dependencies resolved by the linker.

- **Archive files/Static libraries**: Static libraries are bundled as archive files, which are collections of object files grouped together for easy reuse and distribution. Archive files have a global header with additional metadata, followed by the \ac{ELF}-formatted object files. When a program is linked against a static library, the necessary object code is copied into the final executable.

## Software reverse engineering

Software reverse engineering is a systematic process of analyzing and understanding how a program works without access to its original source code or internal documentation. At its core, reverse engineering involves working backwards from a compiled program to comprehend its functionality, architecture, and behavior - the opposite direction of traditional software development. Reverse engineering has its origins in hardware reverse engineering, where analysis of competitors designs was used to gain a competitive advantage. Software reverse engineering is primarily used today for understanding program behavior instead of replication. Whether investigating potentially malicious code or maintenance of legacy systems, software reverse engineering provides insights when source code and documentation are unavailable [@Chikofsky1990; @Fauzi2017; @Muller2009; @Qasem2022].

Software reverse engineering serves many purposes in the digital landscape of today. In the domain of cybersecurity, it enables many types of vulnerability detection, where security researchers and bug hunters identify exploitable pieces of code. It can also be used to identify and analyze malware, protecting critical systems from infected executables and preventing cyberattacks [@Ding2019; @Subedi2018; @Votipka2020; @Qasem2022]. Beyond cybersecurity, reverse engineering enables software interoperability by allowing engineers to understand how systems interact when documentation is unavailable. It can play a vital role in software maintenance, especially for legacy systems where original documentation or development expertise has been lost. Software reverse engineering also serves important legal and compliance functions, helping organizations verify adherence to security standards and licensing requirements. It can also support digital forensics through code similarity detection and ownership attribution [@Votipka2020; @Muller2009; @Qasem2022; @Shoshitaishvili2016; @Fauzi2017; @Luo2014; @Popov2007].

While software reverse engineering encompasses a broad range of activities beyond the scope of this thesis, understanding the complete process provides context for our research contribution. Although we focus specifically on binary reverse engineering at the lowest level, presenting the entire reverse engineering workflow helps position our work within the larger field. The reverse engineering landscape can be viewed through two perspectives: first, the cognitive strategies and approaches reverse engineers employ when analyzing programs, and second, the practical tools and transformations they use to facilitate this analysis. In the following sections, we explore both the methodical process reverse engineers follow and some of the tools that enable their work at different levels of code abstraction.

### Typical RE process

The thought process of a reverse engineer is often iterative and exploratory, as they try to understand the program's behavior and functionality [@Muller2009; @Qasem2022]. Votipka et al. [@Votipka2020] conducted a survey of reverse engineers and found that the most common high level steps in the reverse engineering process can be grouped into 3 phases:

1. **Overview** \
   The reverse engineer try to establish a high-level understanding of the program. Some reverse engineers report that programs subject for analysis comes with some information, which help point the analysis in the right direction. A common strategy is to list strings used by the programs, which often also points to external API calls. They also look at loaded resources, libraries and try to identify important functions and code segments.
2. **Subcomponent Scanning** \
   In this phase, the reverse engineer scans through prioritized functions and code sections identified in the Overview step. In this scan the reverse engineer looks for so-called beacons; important nuggets of information like API-calls, strings, left over symbol-names from compilation, control-flow structures et cetera. If a function outline is found, the reverse engineer will try to identify the input and output of the function, and how it interacts with other subroutines. Some common algorithms, loops and datastructures can be recognized from experience, and marked for future analysis.
3. **Focused Experimentation** \
   Here the reverse engineer test specific hypotheses through program execution. This can be done by running the program in a debugger to examine memory states, manipulating the execution environment (like registry values or binary patching), comparing to known implementations of suspected algorithms, and when necessary reading code line-by-line for detailed understanding. Another valuable strategy is fuzzing; varying inputs to subcomponents and test against expected changes to the output. The results from this phase are then fed back into the subcomponent scanning phase for iterative refinement.

The reverse engineering process is often iterative and exploratory, and the reverse engineers have different approaches depending on the problem. A lot of the strategies used comes from experience and intuition, and the reverse engineer has to adapt their strategy to the specific program. This makes it difficult to create a one-size-fits-all approach to reverse engineering. The process is often not linear, and the frequently reverse engineer jumps between phases and tasks [@Votipka2020; @Muller2009]. However, a key theme across the sources is the need to separate the program into bite-sized blocks, hypothesize how they work, and test that hypothesis with experimentation. Another important strategy is to visualize the program flow, like drawing out the control flow graph or how different subcomponents connect to each other. This helps the reverse engineer understand how different components interact and how data flows through the program [@Muller2009; @Qasem2022].

### Tools and challenges

In order for reverse engineers to analyze a program the code needs to be in a human-readable format. Software reverse engineering is reliant on tools that transform programs in to digestible forms, like binary code to assembly, or assembly to higher level abstractions. Each level of abstraction comes with unique challenges that might need to be overcome in order to apply the tools and reverse engineer the program.

#### Binary reverse engineering and disassemblers

At the lowest level, when presented with a binary of unknown origin, reverse engineers use _disassemblers_ like objdump, angr and IDA Pro along with obtained knowledge of the \ac{ISA} to translate the binary into assembly instructions [@idapro; @angr; @GorkeSteensland2024]. Metadata about the \ac{ISA} and target system is usually present in binary file headers like \ac{ELF}, making disassembly a quite simple for known architectures. The main challenges at this level are figuring out the \ac{ISA} if it is unknown or undocumented, as well as identifying code sections, program entry point and function boundaries so that execution can be followed. Some binaries are also be compressed or encrypted, also inhibiting disassembly [@GorkeSteensland2024; @Kairajarvi2020; @Nicolao2018; @Qasem2022].

#### Decompilers and higher level analysis

While assembly captures the semantic meaning of a program in a more human-readable manner and smaller pieces of code can be analyzed by reverse engineers, larger software systems are often too complex for meaningful information to be extracted purely through disassembly. _Decompilers_ are one such tool to aid in obtaining higher level understanding of the software. Decompilers like IDA Pro, angr and Ghidra use assembly to reconstruct the program in a higher level language like C to improve human readability and make program semantics easier to understand [@idapro; @angr; @ghidra]. However, inherent limitations with information loss during compilation makes it virtually impossible to reconstruct the original source from assembly. Software developers rely on variable names and code comments to document data structures and code, which are often lost during release builds of the software. Modern compilers also perform performance or memory optimizations like loop unrolling, function inlining, changing arithmetic operands and control flow optimizations that can significantly transform the original code structure, making it even more challenging to map between source code and the resulting assembly [@GorkeSteensland2024; @Qasem2022; @Votipka2020].

In addition to disassembly, some tools are able to lift binaries into higher level representations with semantic analysis or like language-invariant intermediate representations such as LLVM IR. BinJuice is one such semantic analysis, which tries to capture program state changes performed by code blocks [@BinJuice2013]. While intermediate representations are typically used as a steppingstone in compilation and decompilation, their language-agnostic nature makes them valuable for large-scale program analysis. By also comparing code at the intermediate representation level, analysts enhance code similarity detection and semantic analysis, which can help enable understanding of program behavior and structure [@GorkeSteensland2024; @Qasem2022].

#### Obfuscation

Obfuscation is a technique that aims to make reverse engineering more difficult, by transforming the code in a way that preserves its functionality but makes it harder to understand. This can be achieved through various methods, such as manipulating the control flow of the program in order to make execution harder to follow, alter common data structure layouts and strings, and changing the layout of the file itself and order of instructions. Tools like Tigress and Obfuscator-LLVM can take in a working program and apply these transformations automatically [@tigress; @ollvm2015]. Obfuscation can be used to protect the intellectual property of the program, make finding and exploiting bugs harder, all by deterring reverse engineering efforts. It can also be used maliciously like circumventing malware detection tools. Obfuscation can make reverse engineering more challenging, however as the program is semantically equivalent, it is still possible to analyze the program and understand its behavior [@Ding2019; @Luo2014; @Popov2007].

### What is needed to reverse engineer

TODO

- Explain what ISA features are needed for reverse engineering

## Machine learning

### Deep learning

Modern machine learning has roots all the way back to the 1950s, when the first artificial neural network was implemented. The term machine learning was introduced around the same time. The development of neural networks continued during the 60s and 70s alongside statistical learning methods such as the nearest neighbor and decision tree methods. However, advances in neural networks faced challenges due to research that demonstrated fundamental limitations of single-layer networks. In the 1980s, the backpropagation algorithm (which is still widely used today) was popularized, and solved the problem of training multi-layer networks. This put neural networks back on the map, and the field continued advancing through the 90s and 2000s. Significant breakthroughs such as parallelized training pipelines, generational leaps in computing power, and utilization of GPUs for machine learning tasks accelerated deep learning developments. This led up to a turning point in 2012, when AlexNet, a deep convolutional neural network, won the annual ImageNet competition. Since then, deep learning approaches have dominated the field of machine learning.

### Convolutional neural networks

\acp{CNN} is a deep learning technique designed for processing grid-based data. It is most commonly applied to visual tasks such as image classification and object detection. The main invention of \acp{CNN} is the concept of convolution layers. These layers scan across the input using _kernels_. The kernels detect features such as edges, textures, and patterns in the input data, and each output a feature map that is passed to the next layer. Each kernel has parameters that are trained based on the entire input grid. \autoref{fig:sliding-kernel} shows an example of a kernel sliding over the input.

![A 3x3 kernel sliding over a 4x4 input. This layer will result in a 2x2 output. \label{fig:sliding-kernel}](images/background/sliding-kernel.svg)

Most \ac{CNN} architectures also use pooling layers, which are static, non-trainable layers that reduce the spatial dimensions of the data. Activation layers, usually ReLU, are used to introduce non-linearity into the network. Finally, fully-connected layers at the end of the network are used for final classification. \autoref{fig:cnn-architecture} shows an example of a basic \ac{CNN} architecture.

![Simple CNN architecture \label{fig:cnn-architecture}](images/background/cnn-architecture.svg)

\acp{CNN} provide several advantages over competing approaches:

- Where traditional computer vision methods usually require significant feature engineering efforts, a \ac{CNN} is able to automatically detect and learn features from the input data without manual feature extraction. This saves time and effort, and even enables models to detect patterns that human intuition would be unable to.

- \ac{CNN} is more computationally efficient than fully-connected neural networks. Where fully-connected networks need parameters for every single connection between neurons, a \ac{CNN} uses the same kernels across the entire input, which dramatically reduces the number of trainable parameters. Additionally, the nature of \acp{CNN} make them more feasible for parallelization, better utilizing specialized hardware such as GPUs.

- \ac{CNN} models are _translation invariant_. This means that they can recognize objects, patterns, and textures regardless of their spatial position in the input. This makes the models more versatile and generalizable than fully-connected neural networks.

Hundreds of different \ac{CNN} architectures have been proposed in previous literature. LeNet-5, which is considered the first modern CNN architecture, has around 60 000 trainable parameters [@Lecun98]. Today, large-scale \ac{CNN} architectures such as VGG-16 often have over 100 million trainable parameters [@Simonyan2015].

Choosing a \ac{CNN} architecture is often a trade-off between several factors:

- **Dataset size**: In general, more complex models require larger datasets. In cases where training data is limited, smaller architectures should be considered. Small dataset sizes combined with complex networks often lead to overfitting, meaning the model matches the training data so well that it fails to generalize to unseen data.

- **Training resources**: Larger models are more expensive to train in terms of computation power. Training deep learning models efficiently often requires use of powerful \acp{GPU}.

- **Inference resources**: Larger models do not only increase the cost of training, it also increases the cost of inference, i.e. making predictions using the trained model. Depending on where the model will be deployed, this may be a deciding factor.

### Overfitting and regularization

When training machine learning models, especially when model complexity is high, there is a risk of overfitting. Overfitting occurs when a model learns the training data too perfectly, including its noise and random fluctuations, rather than learning the true underlying patterns. A model that is overfit to the training data will fail to generalize to unseen data, which causes weak performance in real-world applications.

\autoref{fig:overfitting} shows an example of performance behavior of a model that is overfitting. We see that both training accuracy and validation accuracy improves initially, but after 7 epochs, the validation accuracy stagnates while training accuracy keeps increasing.

![Example of training and validation accuracy when overfitting \label{fig:overfitting}](./images/background/overfitting.svg)

Regularization is a set of techniques for reducing overfitting in machine learning models. In general, regularization penalizes models that are very complex, which incentivizes simpler models that likely generalize better to unseen data.

#### Ridge and Lasso regularization

Lasso (L1) regularization and Ridge (L2) regularization both work by appending a regularization term to the loss function. The standard loss function for Mean Squared Error (MSE) is:

$$
L_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2
$$

Lasso (L1) regularization sums up the absolute value of all model weights, multiplies it by a regularization strength $\lambda$, and adds this term to the loss function:

$$
L_{lasso} = \underbrace{\frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2}_{\text{MSE}} + \underbrace{\lambda\sum_{j=1}^p|w_j|}_{\text{L1 penalty}}
$$

Ridge (L2) regularization sums up the absolute value of all model weights, multiplies it by a regularization strength $\lambda$, and adds this term to the loss function:

$$
L_{ridge} = \underbrace{\frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2}_{\text{MSE}} + \underbrace{\lambda\sum_{j=1}^pw_j^2}_{\text{L2 penalty}}
$$

Since the training process tries to update the model parameters in such a way that the loss function is minimized, both these regularization techniques result in models that balance fitting on the training data with maintaining simplicity.

#### Weight decay

Where L1 and L2 regularization add a regularization term to the loss function, weight decay instead modifies the weight update rule to include a decay factor. The standard weight update rule for gradient descent is:

$$
w_t = w_{t-1} - \eta\frac{\partial L}{\partial w}
$$

Weight decay scales down the previous parameter value by a factor of $(1 - \eta\lambda)$, where $\eta$ is the learning rate and $\lambda$ is the weight decay coefficient:

$$
w_t = w_{t-1}\underbrace{(1 - \eta\lambda)}_{\text{Decay factor}} - \eta\frac{\partial L}{\partial w}
$$

This causes the weights to gradually decay unless the gradient update is large enough to counteract it, leading to simpler models with smaller weights.

When using standard gradient descent for training, weight decay is mathematically equivalent to Ridge (L2) regularization. However, with adaptive optimizers such as Adam, L2 regularization gets scaled by the adaptive learning rates. Weight decay avoids this issue by applying the penalty directly to the weights, which makes it preferable to L2 regularization for these optimizers.

#### Dropout

Dropout is a regularization technique that randomly deactivates a portion of neurons during each training iteration. During training, each neuron has a probability $p$ of being temporarily removed from the network along with its connections. For the next iteration, all neurons are restored before randomly dropping a new subset.

During testing and inference, no neurons are dropped. Instead, all neuron outputs are scaled by $p$ to maintain the same expected magnitude of activations:

$$
h_{\text{test}} = p \cdot h
$$

By forcing the network to function without access to all neurons during training, dropout prevents any neuron from relying too heavily on a specific subset of other neurons. This encourages the network to learn redundant representations, which improves robustness. The technique is particularly effective for very deep networks, where common dropout probabilities range from 0.2 to 0.5.

<!-- TODO: explain K-fold and LOGO? -->

### Leave-one-group-out cross validation

_Cross validation_ is a technique used to assess performance and generalizability of a machine learning model. It involves partitioning data into subsets, where the model is trained on certain subsets while validated using the remaining ones. The process is repeated, making sure the model is trained and validated using different splits. This helps reduce overfitting on a fixed validation set, with the trade-off of requiring more computation since the model needs to be trained multiple times. \autoref{fig:cross-validation} illustrates how the data can be split in a 5-fold cross validation. Each fold serves as validation data once while remaining data is used for training.

It is worth noting that cross validation is used only when verifying the model architecture and hyperparameters, not when training the actual model that will be deployed. After performance is assessed using cross validation techniques, the final model is trained on all available training data without holding out a validation set.

![K-fold cross validation with 5 folds. \label{fig:cross-validation}](images/background/cross-validation.svg)

\ac{LOGO CV} is a variation used in cases where the data is grouped into distinct clusters or categories. The purpose is to ensure that the trained model is tested on data independent of the groups it was trained on. Instead of partitioning the data into random subsets, it is split into groups based on a predefined variable. For each iteration, one group is left out as the validation set, and the model is trained on the remaining groups. This technique assesses how well the model generalizes to completely unseen groups, which is especially useful in case the final model will be used with data that does not belong to any of the groups present in the training data.

### Embeddings

Embeddings are a way a to convert discrete data like words, categories, or items into continuous vectors of numbers that capture semantic relationships or patterns. These vectors are part of the model's trainable parameters, which allows the model to discover semantics from the input data during training. It is common to include an embedding layer as the first layer of a deep learning model, essentially creating a mathematical representation of arbitrary categorical data.

Word embeddings serve as the foundation for many natural language processing tasks. When trained on massive English datasets, these embeddings often capture sophisticated semantic relationships. A classic example in the literature demonstrates this through vector arithmetic: starting with the vector of "king", subtracting "man", and adding "woman", we end up with a vector that is very close to that of "queen" [@Mikolov2013].

### Transfer learning

Transfer learning is the concept of training a model for one purpose, and then re-using some or all of the trained weights for a different application. For instance, CNN models trained on ImageNet – a large dataset of 3.2 million images – are readily available online [@ImageNet]. Transfer learning is useful in cases where there is not enough training data to train a model from scratch, or when there are limited amounts of time or computational resources available.

#### Fine-tuning

Fine-tuning involves taking a pre-trained model and then training it further on a new but related task. The model's weights are updated to work with the new task, while retaining knowledge from pre-training. Common approaches include full fine-tuning (updating all parameters), partial fine-tuning (updating certain layers while others reming frozen), and gradual fine-tuning (progressively increasing the learning rate across layers).

#### Feature extraction

Instead of updating the model's weights, we can use the pre-trained model as a feature extractor. In this case, we typically remove the final classification layers of the pre-trained model and feed the remaining layers into a new classifier that is trained on our own data. The pre-trained layers remain completely frozen during training.

## Related work

In a specialization project completed prior to this thesis [@Preproject], we conducted a systematic literature review exploring two avenues of existing research:

1. Applying machine learning techniques for \ac{ISA} detection
2. Applications of CNN for binary machine code

Subsections \ref{machine-learning-for-isa-detection} and \ref{cnn-applications-for-binary-machine-code} summarize our findings from this systematic literature review. Note that parts of the specialization project is revised and reused in these subsections.

Finally, in \autoref{classifying-isa-features-with-machine-learning}, we describe a master thesis from 2024 researching traditional (non-deep) machine learning for detecting specific architectural features from unknown or undocumented \acp{ISA}.

### Machine learning for ISA detection

TODO

### CNN applications for binary machine code

Although \acp{CNN} are traditionally applied to vision tasks such as image classification or object recognition, prior research has also used \acp{CNN} for analyzing binary machine code. Notably, research on malware detection and classification has proven that \acp{CNN} can perform well on raw binary code.

#### Malware classification

The \ac{MMCC} dataset was published as part of a research competition in 2015, and contains malware binaries from 9 different malware families [@MMCC]. Multiple researchers have used this dataset and developed \ac{CNN} architectures for distinguishing the different classes of malware. A summary of said literature and their classification performance is shown in \autoref{table:mmcc-results}.

Table: Microsoft Malware dataset classification performance. \label{table:mmcc-results}

| Paper (year published)                  | Accuracy   | Precision  | Recall     | F1-score   |
| --------------------------------------- | ---------- | ---------- | ---------- | ---------- |
| Rahul et al. [@Rahul2017] (2017)        | 0.9491     | -          | -          | -          |
| Kumari et al. [@Kumari2017] (2017)      | 0.9707     | -          | -          | -          |
| Yang et al. [@Yang2018] (2018)          | 0.987      | -          | -          | -          |
| Khan et al. [@Khan2020] (2020)          | 0.9780     | 0.98       | 0.97       | 0.97       |
| Sartoli et al. [@Sartoli2020] (2020)    | 0.9680     | 0.9624     | 0.9616     | 0.9618     |
| Bouchaib & Bouhorma [@Prima2021] (2021) | 0.98       | 0.98       | 0.98       | 0.98       |
| Liang et al. [@Liang2021] (2021)        | 0.9592     | -          | -          | -          |
| SREMIC [@Alam2024] (2024)               | **0.9972** | **0.9993** | **0.9971** | **0.9988** |

Malimg is another dataset containing malware from 25 different families [@Malimg]. As opposed to \ac{MMCC}, this dataset contains binaries pre-encoded to an image format, using each byte in the file as a single pixel value. A summary of identified research applying \acp{CNN} to this dataset is shown in \autoref{table:malimg-results}.

Table: Malimg dataset classification performance. \label{table:malimg-results}

| Paper (year published)                   | Accuracy   | Precision  | Recall     | F1-score   |
| ---------------------------------------- | ---------- | ---------- | ---------- | ---------- |
| Cervantes et al. [@Garcia2019] (2019)    | 0.9815     | -          | -          | -          |
| El-Shafai et al. [@El-Shafai2021] (2021) | **0.9997** | 0.9904     | 0.9901     | 0.9902     |
| Li et al. [@Li2021] (2021)               | 0.97       | -          | -          | -          |
| Son et al. [@Son2022] (2022)             | 0.97       | -          | -          | -          |
| Hammad et al. [@Hammad2022] (2022)       | 0.9684     | -          | -          | -          |
| S-DCNN [@Parihar2022] (2022)             | 0.9943     | 0.9944     | 0.9943     | 0.9943     |
| SREMIC [@Alam2024] (2024)                | 0.9993     | **0.9992** | **0.9987** | **0.9987** |
| DCMN [@Al-Masri2024] (2024)              | 0.9989     | 0.9971     | 0.9984     | 0.9977     |

The existing literature differs in their approach to data encoding and model architecture. Certain studies used a one-dimensional vector encoding of the binary data, where each byte in the binary file is converted to a decimal value between 0 and 255. The encoded bytes were then passed to a one-dimensional \ac{CNN} for classification [@Rahul2017] [@Li2021]. However, the most common approach was to encode the binary data as a two-dimensional image, where each byte in the binary file is converted to a pixel value. These grayscale images are then used for input to traditional two-dimensional \ac{CNN} architectures [@Kumari2017] [@Prima2021] [@Hammad2022] [@Al-Masri2024] [@Yang2018] [@El-Shafai2021] [@Alvee2021] [@Liang2021] [@Son2022].

#### Compiler optimization detection

Compilers such as GCC allow the user to choose between five general optimization levels: -O0, -O1, -O2, -O3, and -Os. Knowing which of these levels was used for compilation can be useful in areas such as vulnerability discovery. We identified two prior studies that attempt detecting a binary's compiler optimization level.

Yang et al. achieved an overall accuracy of 97.24% on their custom dataset, with precision for each class ranging from 96% to 98% [@Yang2019]. This was a significant improvement compared to previous literature regarding compiler level discovery. Pizzolotto & Inoue elaborated on this work by using binaries compiled across 7 different CPU architectures, as well as compiling with both GCC and Clang for the x86-64 and AArch64 architectures [@Pizzolotto2021]. They showed a 99.95% accuracy in distinguishing between GCC and Clang, while the optimization level accuracy varies from 92% to 98% depending on the CPU architecture. However, we note that Pizzolotto & Inoue treated -O2 and -O3 as separate classes, whereas Yang et al. considered these as the same class, making the comparison slightly unfair.

Both studies used a one-dimensional vector encoding of the binary data, where each byte in the binary file is converted to a decimal value between 0 and 255. The encoded bytes were then passed to a one-dimensional \ac{CNN} for classification.

### Classifying ISA features with machine learning

Joachim Andreassen's master's thesis from 2024 explored the detection of architectural features from binary programs where the \ac{ISA} is unknown or undocumented [@Andreassen_Morrison_2024]. Unlike the studies discussed in \autoref{machine-learning-for-isa-detection}, which applied machine learning techniques to \ac{ISA} detection, Andreassen attempted to detect endianness and instruction width from binaries that come from previously unseen \acp{ISA}.

The methods applied in this thesis are traditional, non-deep machine learning models such as random forests, k-nearest neighbors, and logistic regression. It makes heavy use of feature engineering, using both existing features from prior literature as well as engineering novel features, particularly for detecting instruction width. Some of these custom features use signal processing techniques such as autocorrelation and Fourier transformation.

Using \acf{LOGO CV} on the ISADetect dataset, training on only the code section part of the binary file, the best-performing models achieve an average accuracy of 92.0% on endianness detection and 86.0% in distinguishing between fixed and variable instruction width. We will use Andreassen's thesis as a comparison for determining whether our \ac{CNN} approach can eliminate the need for manual feature engineering, without sacrificing model performance.
