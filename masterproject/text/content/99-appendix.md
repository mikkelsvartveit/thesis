\appendix
\acresetall

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

Table: The list of labels used in the BuildCross dataset. The labels are based on the ELF headers of the generated code and the disassembly of the binaries \label{table:buildcross-labels-full}

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

## ISAdetect dataset label list and sizes

Table: The list of labels and architecture data sizes in the ISAdetect dataset. The labels are based on the labeling by Kairajärvi et al. [@Kairajarvi_dataset2020] \label{table:isadetect-labels-full}

| Architecture | Endianness | Instruction Width Type | Total Size (MB) | Number of Files |
| ------------ | ---------: | ---------------------: | --------------: | --------------: |
| alpha        |     little |                  fixed |          925.77 |            3952 |
| amd64        |     little |               variable |          564.43 |            4059 |
| arm64        |     little |                  fixed |          418.50 |            3518 |
| armel        |     little |                  fixed |          466.91 |            3814 |
| armhf        |     little |                  fixed |          331.34 |            3674 |
| hppa         |        big |                  fixed |          940.76 |            4830 |
| i386         |     little |               variable |          519.50 |            4484 |
| ia64         |     little |               variable |         2044.75 |            4983 |
| m68k         |        big |               variable |          684.06 |            4313 |
| mips         |        big |                  fixed |          545.13 |            3547 |
| mips64el     |     little |                  fixed |         1117.75 |            4280 |
| mipsel       |     little |                  fixed |          545.68 |            3693 |
| powerpc      |        big |                  fixed |          547.82 |            3618 |
| powerpcspe   |        big |                  fixed |          790.11 |            3922 |
| ppc64        |        big |                  fixed |          771.06 |            2822 |
| ppc64el      |     little |                  fixed |          574.67 |            3521 |
| riscv64      |     little |                  fixed |          605.14 |            4285 |
| s390         |        big |               variable |          360.46 |            5118 |
| s390x        |        big |               variable |          532.86 |            3511 |
| sh4          |     little |                  fixed |          723.25 |            5854 |
| sparc        |        big |                  fixed |          362.99 |            4923 |
| sparc64      |        big |                  fixed |          844.07 |            3205 |
| x32          |     little |               variable |          719.76 |            4059 |

## CpuRec dataset label list and sizes

Table: The list of labels used in the CpuRec dataset. The labels are based on previous work by [@Andreassen_Morrison_2024], searching online for ISA documentation and disassembly from the BuildCross suite. \label{table:cpurec-labels-full}

| Architecture | Endianness | Instruction Width Type | Total Size (KB) |
| ------------ | ---------- | ---------------------- | --------------- |
| 6502         | little     | variable               | 6.57            |
| 68HC08       | big        | variable               | 18.39           |
| 68HC11       | big        | variable               | 25.17           |
| 8051         | unknown    | variable               | 15.76           |
| ARC32eb      | big        | variable               | 46.09           |
| ARC32el      | little     | variable               | 45.88           |
| ARM64        | little     | fixed                  | 345.32          |
| ARMeb        | big        | fixed                  | 896.44          |
| ARMel        | little     | fixed                  | 329.23          |
| ARMhf        | little     | fixed                  | 230.78          |
| ARcompact    | little     | variable               | 118.12          |
| AVR          | unknown    | variable               | 193.40          |
| Alpha        | little     | fixed                  | 1065.17         |
| AxisCris     | little     | variable               | 61.11           |
| Blackfin     | little     | variable               | 104.82          |
| CLIPPER      | little     | variable               | 1059.47         |
| Cell-SPU     | unknown    | unknown                | 290.22          |
| CompactRISC  | little     | variable               | 56.58           |
| Cray         | unknown    | variable               | 1120.00         |
| Epiphany     | little     | variable               | 69.03           |
| FR-V         | big        | fixed                  | 175.25          |
| FR30         | big        | fixed                  | 141.42          |
| FT32         | little     | fixed                  | 179.18          |
| H8-300       | big        | variable               | 163.47          |
| H8S          | unknown    | variable               | 81.52           |
| HP-Focus     | unknown    | variable               | 408.00          |
| HP-PA        | big        | fixed                  | 1057.03         |
| IA-64        | little     | variable               | 423.41          |
| IQ2000       | big        | fixed                  | 178.65          |
| M32C         | little     | variable               | 173.75          |
| M32R         | big        | fixed                  | 121.89          |
| M68k         | big        | variable               | 728.19          |
| M88k         | big        | fixed                  | 351.18          |
| MCore        | little     | fixed                  | 101.30          |
| MIPS16       | unknown    | fixed                  | 95.62           |
| MIPSeb       | big        | fixed                  | 747.85          |
| MIPSel       | little     | fixed                  | 425.16          |
| MMIX         | big        | fixed                  | 387.71          |
| MN10300      | little     | variable               | 114.29          |
| MSP430       | little     | variable               | 301.51          |
| Mico32       | big        | fixed                  | 163.39          |
| MicroBlaze   | big        | fixed                  | 192.71          |
| Moxie        | big        | variable               | 140.64          |
| NDS32        | little     | variable               | 94.04           |
| NIOS-II      | little     | fixed                  | 139.11          |
| PDP-11       | unknown    | variable               | 124.00          |
| PIC10        | unknown    | fixed                  | 8.89            |
| PIC16        | unknown    | fixed                  | 39.16           |
| PIC18        | unknown    | fixed                  | 45.89           |
| PIC24        | little     | fixed                  | 82.67           |
| PPCeb        | big        | fixed                  | 403.82          |
| PPCel        | little     | fixed                  | 462.20          |
| RISC-V       | little     | fixed                  | 69.24           |
| RL78         | little     | variable               | 337.46          |
| ROMP         | big        | variable               | 440.00          |
| RX           | little     | variable               | 87.12           |
| S-390        | big        | variable               | 453.77          |
| SPARC        | big        | fixed                  | 1376.51         |
| STM8         | unknown    | variable               | 15.35           |
| Stormy16     | little     | variable               | 138.34          |
| SuperH       | little     | fixed                  | 876.43          |
| TILEPro      | unknown    | variable               | 112.16          |
| TLCS-90      | unknown    | variable               | 23.18           |
| TMS320C2x    | unknown    | variable               | 44.94           |
| TMS320C6x    | unknown    | fixed                  | 105.53          |
| TriMedia     | unknown    | unknown                | 462.70          |
| V850         | little     | variable               | 132.65          |
| VAX          | little     | variable               | 318.00          |
| Visium       | big        | fixed                  | 274.00          |
| WE32000      | unknown    | unknown                | 326.32          |
| X86          | little     | variable               | 396.49          |
| X86-64       | little     | variable               | 375.41          |
| Xtensa       | unknown    | variable               | 87.56           |
| XtensaEB     | big        | variable               | 66.03           |
| Z80          | little     | variable               | 20.86           |
| i860         | unknown    | fixed                  | 598.00          |

## Dataset labeling comparison with Andreassen

Table: ISAdetect labeling differences between our research and what was presented in Andreassen's paper. Differences highlighted in bold. \label{table:isadetect-labels-comparison}

| ISA        | Our endianness | Andreassen endianness | Our instruction width type | Andreassen instruction width type |
| ---------- | -------------: | :-------------------- | -------------------------: | :-------------------------------- |
| alpha      |         little | little                |                      fixed | fixed                             |
| amd64      |         little | little                |                   variable | variable                          |
| arm64      |         little | little                |                      fixed | fixed                             |
| armel      |         little | little                |                      fixed | fixed                             |
| armhf      |         little | little                |                      fixed | fixed                             |
| hppa       |            big | big                   |                      fixed | fixed                             |
| i386       |         little | little                |                   variable | variable                          |
| ia64       |         little | little                |               **variable** | **fixed**                         |
| m68k       |            big | big                   |               **variable** | **unk**                           |
| mips       |            big | big                   |                      fixed | fixed                             |
| mips64el   |         little | little                |                      fixed | fixed                             |
| mipsel     |         little | little                |                      fixed | fixed                             |
| powerpc    |            big | big                   |                      fixed | fixed                             |
| powerpcspe |            big | big                   |                      fixed | fixed                             |
| ppc64      |            big | big                   |                  **fixed** | **unk**                           |
| ppc64el    |         little | little                |                  **fixed** | **unk**                           |
| riscv64    |         little | little                |                      fixed | fixed                             |
| s390       |            big | big                   |               **variable** | **unk**                           |
| s390x      |            big | big                   |               **variable** | **unk**                           |
| sh4        |     **little** | **BI**                |                  **fixed** | **unk**                           |
| sparc      |            big | big                   |                      fixed | fixed                             |
| sparc64    |            big | big                   |                      fixed | fixed                             |
| x32        |         little | little                |               **variable** | **unk**                           |

Table: CpuRec labeling differences between our research and what was presented in Andreassen's paper. Differences highlighted in bold. 78k was not in the corpus at the time of downloading the dataset. \label{table:cpurec-labels-comparison}

| ISA         | Our Endianness | Andreassen Endianness | Our instruction width type | Andreassen instruction width type |
| :---------- | -------------: | :-------------------- | -------------------------: | :-------------------------------- |
| 6502        |         little | little                |                   variable | variable                          |
| 68HC08      |            big | big                   |                   variable | variable                          |
| 68HC11      |            big | big                   |                   variable | variable                          |
| 78k         |              - |                       |                          - |                                   |
| 8051        |         **na** | **little**            |                   variable | variable                          |
| Alpha       |         little | little                |                      fixed | fixed                             |
| ARCompact   |         little | little                |                   variable | variable                          |
| ARM64       |         little | little                |                      fixed | fixed                             |
| ARMeb       |            big | big                   |                      fixed | fixed                             |
| ARMel       |         little | little                |                      fixed | fixed                             |
| ARMhf       |         little | little                |                      fixed | fixed                             |
| AVR         |         **na** | **little**            |                   variable | variable                          |
| AxisCris    |         little | little                |               **variable** | **fixed**                         |
| Blackfin    |         little | little                |                   variable | variable                          |
| Cell-SPU    |         **bi** | **big**               |                    **unk** | **fixed**                         |
| CLIPPER     |         little | little                |                   variable | variable                          |
| CompactRISC |         little | little                |               **variable** | **fixed**                         |
| Cray        |             na | NA                    |                   variable |                                   |
| Epiphany    |         little | little                |                   variable | variable                          |
| FR-V        |            big | NA                    |                      fixed |                                   |
| FR30        |            big | big                   |                      fixed | fixed                             |
| FT32        |         little | NA                    |                      fixed |                                   |
| H8-300      |            big | big                   |                   variable | variable                          |
| H8S         |            unk | big                   |                   variable |                                   |
| HP-Focus    |             na | NA                    |                   variable |                                   |
| HP-PA       |            big | big                   |                      fixed | fixed                             |
| i860        |             bi | BI                    |                      fixed |                                   |
| IA-64       |         little | little                |               **variable** | **fixed**                         |
| IQ2000      |            big | big                   |                      fixed |                                   |
| M32C        |         little | NA                    |                   variable |                                   |
| M32R        |        **big** | **BI**                |                  **fixed** | **variable**                      |
| M68k        |            big | big                   |                   variable |                                   |
| M88k        |        **big** | **BI**                |                      fixed | fixed                             |
| MCore       |     **little** | **big**               |                      fixed | fixed                             |
| Mico32      |            big | big                   |                      fixed | fixed                             |
| MicroBlaze  |        **big** | **BI**                |                      fixed | fixed                             |
| MIPS16      |             bi | BI                    |                      fixed | fixed                             |
| MIPSeb      |            big | big                   |                      fixed | fixed                             |
| MIPSel      |         little | little                |                      fixed | fixed                             |
| MMIX        |            big | big                   |                      fixed | fixed                             |
| MN10300     |         little | little                |                   variable |                                   |
| Moxie       |        **big** | **BI**                |                   variable | variable                          |
| MSP430      |         little | little                |                   variable |                                   |
| NDS32       |     **little** | **BI**                |                   variable | variable                          |
| NIOS-II     |         little | little                |                      fixed | fixed                             |
| PDP-11      |     **middle** | **little**            |               **variable** | **fixed**                         |
| PIC10       |         **na** | **little**            |                      fixed |                                   |
| PIC16       |         **na** | **little**            |                      fixed |                                   |
| PIC18       |         **na** | **little**            |                      fixed |                                   |
| PIC24       |         little | little                |                      fixed | fixed                             |
| PPCeb       |            big | big                   |                      fixed |                                   |
| PPCel       |         little | little                |                      fixed |                                   |
| RISC-V      |         little | little                |                      fixed | fixed                             |
| RL78        |         little | little                |                   variable |                                   |
| ROMP        |            big | big                   |                   variable | variable                          |
| RX          |         little | little                |                   variable |                                   |
| S-390       |            big | big                   |                   variable |                                   |
| SPARC       |            big | big                   |                      fixed | fixed                             |
| STM8        |             na |                       |                   variable |                                   |
| Stormy16    |         little | little                |                   variable |                                   |
| SuperH      |     **little** | **BI**                |                      fixed |                                   |
| TILEPro     |            unk |                       |                   variable |                                   |
| TLCS-90     |            unk |                       |                   variable |                                   |
| TMS320C2x   |            unk |                       |                   variable |                                   |
| TMS320C6x   |        **unk** | **BI**                |                      fixed |                                   |
| TriMedia    |            unk |                       |                        unk |                                   |
| V850        |         little |                       |                   variable |                                   |
| Visium      |            big |                       |                      fixed |                                   |
| WE32000     |            unk |                       |                        unk |                                   |
| X86-64      |         little | little                |                   variable | variable                          |
| X86         |         little | little                |                   variable | variable                          |
| Xtensa      |             bi | BI                    |                   variable | variable                          |
| Z80         |         little | little                |                   variable |                                   |

# Statistical analysis material

## Confidence interval and comparison implementation

## Pairwise model comparison

Each model in each evaluation strategy is compared to each other using a paried t-test, testing for whether there are statistically significant differences in model performance. The results are shown in the tables below.

### Endianness

\label{model-comparison-endianness}

![Significance of compared model performance on the kfold endianness evaluation. The p-value refers to the probability, given that there is no significant difference between the two models, that we observe the difference that we have found.](images/appendix/model-comparison/model-comparison-kfold-all-endianness.svg)

![Significance of compared model performance on the \ac{LOGO CV} endianness evaluation. The p-value refers to the probability, given that there is no significant difference between the two models, that we observe the difference that we have found. \label{fig:paired-t-test-logo-endianness}](images/appendix/model-comparison/model-comparison-logo-endianness.svg)

![Significance of compared model performance on the ISAdetect-BuildCross evaluation. The p-value refers to the probability, given that there is no significant difference between the two models, that we observe the difference that we have found. \label{fig:paired-t-test-isadetect-buildcross-endianness}](images/appendix/model-comparison/model-comparison-buildcross-endianness.svg)

![Significance of compared model performance on the ISAdetect-CpuRec evaluation. The p-value refers to the probability, given that there is no significant difference between the two models, that we observe the difference that we have found. \label{fig:paired-t-test-isadetect-cpurec-endianness}](images/appendix/model-comparison/model-comparison-cpurec-endianness.svg)

![Significance of compared model performance on the Combined-CpuRec evaluation. The p-value refers to the probability, given that there is no significant difference between the two models, that we observe the difference that we have found. \label{fig:paired-t-test-combined-cpurec-endianness}](images/appendix/model-comparison/model-comparison-combined-endianness.svg)

\FloatBarrier

### Instruction width type

\label{model-comparison-instructionwidthtype}

![Significance of compared model performance on the K-fold cross validation instruction width type evaluation. The p-value refers to the probability, given that there is no significant difference between the two models, that we observe the difference that we have found.](images/appendix/model-comparison/model-comparison-kfold-all-instructionwidthtype.svg)

![Significance of compared model performance on the \ac{LOGO CV} instruction width type evaluation. The p-value refers to the probability, given that there is no significant difference between the two models, that we observe the difference that we have found.\label{fig:paired-t-test-logo-instructionwidthtype}](images/appendix/model-comparison/model-comparison-logo-instructionwidthtype.svg)

![Significance of compared model performance on the ISAdetect-BuildCross instruction width type evaluation. The p-value refers to the probability, given that there is no significant difference between the two models, that we observe the difference that we have found.\label{fig:paired-t-test-isadetect-buildcross-instructionwidthtype}](images/appendix/model-comparison/model-comparison-buildcross-instructionwidthtype.svg)

![Significance of compared model performance on the ISAdetect-CpuRec instruction width type evaluation. The p-value refers to the probability, given that there is no significant difference between the two models, that we observe the difference that we have found.\label{fig:paired-t-test-isadetect-cpurec-instructionwidthtype}](images/appendix/model-comparison/model-comparison-cpurec-instructionwidthtype.svg)

![Significance of compared model performance on the Combined-CpuRec instruction width type evaluation. The p-value refers to the probability, given that there is no significant difference between the two models, that we observe the difference that we have found.\label{fig:paired-t-test-combined-cpurec-instructionwidthtype}](images/appendix/model-comparison/model-comparison-combined-instructionwidthtype.svg)
