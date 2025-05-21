\appendix

<!-- Reference with \autoref{dataset-additional-information} -->

# Dataset additional information

Contains additional information about the datasets used in this thesis. This includes information about the dataset sizes, number of samples, labeling, as well as comparisons with related work.

## BuildCross dataset label list and sizes

<!-- BuildCross Arch_details:
(Files smaller than 1024 bytes are ignored)
(No Samples are counted in 1024 bytes, with filesplitting)
Total Size: 119.88 MB
Average Size per ISA: 3.00 MB
Median Size per ISA: 2.51 MB -->

The BuildCross dataset contains a total of **119.88 MB** of data, with an average size of **3.00 MB** per ISA and a median size of **2.51 MB** per ISA. The dataset is divided into 40 different ISAs, each with a varying number of files and samples. All files smaller than 1024 bytes are ignored.

Table: The list of labels used in the BuildCross dataset. The labels are based on the elf headers of the generated code and the disassembly of the binaries \label{table:buildcross-labels-full}

| ISA          | Endianness | Instruction Width Type | Total Size (MB) | Number of Files | No. 1024 byte sized samples |
| :----------- | ---------: | ---------------------: | --------------: | --------------: | --------------------------: |
| arc          |     little |               variable |            3.23 |              14 |                        3299 |
| arceb        |        big |               variable |            1.70 |              12 |                        1731 |
| bfin         |     little |               variable |            2.88 |              14 |                        2942 |
| bpf          |     little |                  fixed |            0.02 |               1 |                          19 |
| c6x          |        big |                  fixed |            5.55 |               8 |                        5679 |
| cr16         |     little |               variable |            1.97 |              13 |                        2012 |
| cris         |     little |               variable |            3.98 |              14 |                        4074 |
| csky         |     little |               variable |            4.15 |              14 |                        4247 |
| epiphany     |     little |               variable |            0.46 |               6 |                         471 |
| fr30         |        big |               variable |            2.17 |               7 |                        2223 |
| frv          |        big |                  fixed |            4.93 |              14 |                        5037 |
| ft32         |     little |                  fixed |            0.44 |               9 |                         440 |
| h8300        |        big |               variable |            4.30 |               9 |                        4402 |
| iq2000       |        big |                  fixed |            2.41 |               8 |                        2466 |
| kvx          |     little |               variable |            4.90 |              14 |                        5016 |
| lm32         |        big |                  fixed |            3.32 |              13 |                        3396 |
| loongarch64  |     little |                  fixed |            4.71 |              14 |                        4818 |
| m32r         |        big |                  fixed |            1.96 |              12 |                        1997 |
| m68k-elf     |        big |               variable |            1.83 |              12 |                        1866 |
| mcore        |     little |                  fixed |            1.24 |               7 |                        1270 |
| mcoreeb      |        big |                  fixed |            1.24 |               7 |                        1270 |
| microblaze   |        big |                  fixed |            5.74 |              14 |                        5867 |
| microblazeel |     little |                  fixed |            5.71 |              14 |                        5840 |
| mmix         |        big |                  fixed |            4.22 |              13 |                        4314 |
| mn10300      |     little |               variable |            1.70 |              12 |                        1732 |
| moxie        |        big |               variable |            2.19 |              12 |                        2237 |
| moxieel      |     little |               variable |            2.19 |              12 |                        2232 |
| msp430       |     little |               variable |            0.42 |               5 |                         432 |
| nds32        |     little |               variable |            2.85 |              14 |                        2908 |
| nios2        |     little |                  fixed |            4.21 |              14 |                        4301 |
| or1k         |        big |                  fixed |            5.42 |              14 |                        5544 |
| pru          |     little |                  fixed |            2.39 |               8 |                        2443 |
| rl78         |     little |               variable |            0.63 |               5 |                         643 |
| rx           |     little |               variable |            1.46 |              12 |                        1486 |
| tilegx       |     little |                  fixed |           11.71 |              14 |                       11986 |
| tricore      |     little |               variable |            1.61 |               8 |                        1646 |
| v850         |     little |               variable |            3.53 |              10 |                        3609 |
| visium       |        big |                  fixed |            3.41 |              12 |                        3488 |
| xstormy16    |     little |               variable |            0.48 |               5 |                         490 |
| xtensa       |        big |               variable |            2.61 |              14 |                        2669 |

## Dataset labeling comparison with Andreassen

Table: ISAdetect labeling differences between our research and what was presented in Andreassen's paper. Differences highlighted in bold. \label{table:isadetect-labels-comparison}

| ISA        | Our endianness | Andreassen endianness | Our instructionwidth_type | Andreassen instructionwidth_type |
| ---------- | -------------: | :-------------------- | ------------------------: | :------------------------------- |
| alpha      |         little | little                |                     fixed | fixed                            |
| amd64      |         little | little                |                  variable | variable                         |
| arm64      |         little | little                |                     fixed | fixed                            |
| armel      |         little | little                |                     fixed | fixed                            |
| armhf      |         little | little                |                     fixed | fixed                            |
| hppa       |            big | big                   |                     fixed | fixed                            |
| i386       |         little | little                |                  variable | variable                         |
| ia64       |         little | little                |              **variable** | **fixed**                        |
| m68k       |            big | big                   |              **variable** | **unk**                          |
| mips       |            big | big                   |                     fixed | fixed                            |
| mips64el   |         little | little                |                     fixed | fixed                            |
| mipsel     |         little | little                |                     fixed | fixed                            |
| powerpc    |            big | big                   |                     fixed | fixed                            |
| powerpcspe |            big | big                   |                     fixed | fixed                            |
| ppc64      |            big | big                   |                 **fixed** | **unk**                          |
| ppc64el    |         little | little                |                 **fixed** | **unk**                          |
| riscv64    |         little | little                |                     fixed | fixed                            |
| s390       |            big | big                   |              **variable** | **unk**                          |
| s390x      |            big | big                   |              **variable** | **unk**                          |
| sh4        |     **little** | **BI**                |                 **fixed** | **unk**                          |
| sparc      |            big | big                   |                     fixed | fixed                            |
| sparc64    |            big | big                   |                     fixed | fixed                            |
| x32        |         little | little                |              **variable** | **unk**                          |

Table: CpuRec labeling differences between our research and what was presented in Andreassen's paper. Differences highlighted in bold. 78k was not in the corpus at the time of us downloading the dataset. \label{table:cpurec-labels-comparison}

| ISA         | Our Endianness | Andreassen Endianness | Our instructionwidth_type | Andreassen instructionwidth_type |
| :---------- | -------------: | :-------------------- | ------------------------: | :------------------------------- |
| 6502        |         little | little                |                  variable | variable                         |
| 68HC08      |            big | big                   |                  variable | variable                         |
| 68HC11      |            big | big                   |                  variable | variable                         |
| 78k         |              - |                       |                         - |                                  |
| 8051        |         **na** | **little**            |                  variable | variable                         |
| Alpha       |         little | little                |                     fixed | fixed                            |
| ARCompact   |         little | little                |                  variable | variable                         |
| ARM64       |         little | little                |                     fixed | fixed                            |
| ARMeb       |            big | big                   |                     fixed | fixed                            |
| ARMel       |         little | little                |                     fixed | fixed                            |
| ARMhf       |         little | little                |                     fixed | fixed                            |
| AVR         |         **na** | **little**            |                  variable | variable                         |
| AxisCris    |         little | little                |              **variable** | **fixed**                        |
| Blackfin    |         little | little                |                  variable | variable                         |
| Cell-SPU    |         **bi** | **big**               |                   **unk** | **fixed**                        |
| CLIPPER     |         little | little                |                  variable | variable                         |
| CompactRISC |         little | little                |              **variable** | **fixed**                        |
| Cray        |             na | NA                    |                  variable |                                  |
| Epiphany    |         little | little                |                  variable | variable                         |
| FR-V        |            big | NA                    |                     fixed |                                  |
| FR30        |            big | big                   |                     fixed | fixed                            |
| FT32        |         little | NA                    |                     fixed |                                  |
| H8-300      |            big | big                   |                  variable | variable                         |
| H8S         |            unk | big                   |                  variable |                                  |
| HP-Focus    |             na | NA                    |                  variable |                                  |
| HP-PA       |            big | big                   |                     fixed | fixed                            |
| i860        |             bi | BI                    |                     fixed |                                  |
| IA-64       |         little | little                |              **variable** | **fixed**                        |
| IQ2000      |            big | big                   |                     fixed |                                  |
| M32C        |         little | NA                    |                  variable |                                  |
| M32R        |        **big** | **BI**                |                 **fixed** | **variable**                     |
| M68k        |            big | big                   |                  variable |                                  |
| M88k        |        **big** | **BI**                |                     fixed | fixed                            |
| MCore       |     **little** | **big**               |                     fixed | fixed                            |
| Mico32      |            big | big                   |                     fixed | fixed                            |
| MicroBlaze  |        **big** | **BI**                |                     fixed | fixed                            |
| MIPS16      |             bi | BI                    |                     fixed | fixed                            |
| MIPSeb      |            big | big                   |                     fixed | fixed                            |
| MIPSel      |         little | little                |                     fixed | fixed                            |
| MMIX        |            big | big                   |                     fixed | fixed                            |
| MN10300     |         little | little                |                  variable |                                  |
| Moxie       |        **big** | **BI**                |                  variable | variable                         |
| MSP430      |         little | little                |                  variable |                                  |
| NDS32       |     **little** | **BI**                |                  variable | variable                         |
| NIOS-II     |         little | little                |                     fixed | fixed                            |
| PDP-11      |     **middle** | **little**            |              **variable** | **fixed**                        |
| PIC10       |         **na** | **little**            |                     fixed |                                  |
| PIC16       |         **na** | **little**            |                     fixed |                                  |
| PIC18       |         **na** | **little**            |                     fixed |                                  |
| PIC24       |         little | little                |                     fixed | fixed                            |
| PPCeb       |            big | big                   |                     fixed |                                  |
| PPCel       |         little | little                |                     fixed |                                  |
| RISC-V      |         little | little                |                     fixed | fixed                            |
| RL78        |         little | little                |                  variable |                                  |
| ROMP        |            big | big                   |                  variable | variable                         |
| RX          |         little | little                |                  variable |                                  |
| S-390       |            big | big                   |                  variable |                                  |
| SPARC       |            big | big                   |                     fixed | fixed                            |
| STM8        |             na |                       |                  variable |                                  |
| Stormy16    |         little | little                |                  variable |                                  |
| SuperH      |     **little** | **BI**                |                     fixed |                                  |
| TILEPro     |            unk |                       |                  variable |                                  |
| TLCS-90     |            unk |                       |                  variable |                                  |
| TMS320C2x   |            unk |                       |                  variable |                                  |
| TMS320C6x   |        **unk** | **BI**                |                     fixed |                                  |
| TriMedia    |            unk |                       |                       unk |                                  |
| V850        |         little |                       |                  variable |                                  |
| Visium      |            big |                       |                     fixed |                                  |
| WE32000     |            unk |                       |                       unk |                                  |
| X86-64      |         little | little                |                  variable | variable                         |
| X86         |         little | little                |                  variable | variable                         |
| Xtensa      |             bi | BI                    |                  variable | variable                         |
| Z80         |         little | little                |                  variable |                                  |

# Statistical analysis material

## Confidence interval and comparison implementation

## Pairwise model comparison

Each model in each testing suite is compared to each other using a paried t-test, testing for whether there are statistically significant differences in model performance. The results are shown in the tables below.

![Significance of compared model performance on the kfold endianness suite. The p-value refer to the probability, given that there is no significant difference between the two models, that we see observe the difference that we have found](images/appendix/model-comparison/model-comparison-kfold-all-endianness.svg)

![Significance of compared model performance on the \ac{LOGO CV} endianness suite. The p-value refer to the probability, given that there is no significant difference between the two models, that we see observe the difference that we have found](images/appendix/model-comparison/model-comparison-logo-endianness.svg)

![Significance of compared model performance on the ISAdetect-BuildCross suite. The p-value refer to the probability, given that there is no significant difference between the two models, that we see observe the difference that we have found](images/appendix/model-comparison/model-comparison-buildcross-endianness.svg)

![Significance of compared model performance on the ISAdetect-CpuRec suite. The p-value refer to the probability, given that there is no significant difference between the two models, that we see observe the difference that we have found](images/appendix/model-comparison/model-comparison-cpurec-endianness.svg)

![Significance of compared model performance on the Combined-CpuRec suite. The p-value refer to the probability, given that there is no significant difference between the two models, that we see observe the difference that we have found](images/appendix/model-comparison/model-comparison-combined-endianness.svg)

![Significance of compared model performance on the kfold instructionwidth_type suite. The p-value refer to the probability, given that there is no significant difference between the two models, that we see observe the difference that we have found](images/appendix/model-comparison/model-comparison-kfold-all-instructionwidthtype.svg)

![Significance of compared model performance on the \ac{LOGO CV} instructionwidth_type suite. The p-value refer to the probability, given that there is no significant difference between the two models, that we see observe the difference that we have found](images/appendix/model-comparison/model-comparison-logo-instructionwidthtype.svg)

![Significance of compared model performance on the ISAdetect-BuildCross instructionwidth_type suite. The p-value refer to the probability, given that there is no significant difference between the two models, that we see observe the difference that we have found](images/appendix/model-comparison/model-comparison-buildcross-instructionwidthtype.svg)

![Significance of compared model performance on the ISAdetect-CpuRec instructionwidth_type suite. The p-value refer to the probability, given that there is no significant difference between the two models, that we see observe the difference that we have found](images/appendix/model-comparison/model-comparison-cpurec-instructionwidthtype.svg)

![Significance of compared model performance on the Combined-CpuRec instructionwidth_type suite. The p-value refer to the probability, given that there is no significant difference between the two models, that we see observe the difference that we have found](images/appendix/model-comparison/model-comparison-combined-instructionwidthtype.svg)
