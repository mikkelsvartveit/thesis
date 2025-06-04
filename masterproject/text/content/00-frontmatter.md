---
title: "Unveiling Instruction Set Characteristics from Raw Binary Code using Convolutional Neural Networks"
subtitle: |
  \vspace{0.2cm}
  TDT4900 – Master's Thesis\
  \vspace{0.1cm}
  Fall 2024\
  \vspace{1cm}
  ![NTNU Logo](./images/ntnu.png){ width=12% }\
  \vspace{0.3cm}
  \small
  Norwegian University of Science and Technology\
  Supervised by Donn Morrison\
  \vspace{0.3cm}
date: |
  \today
  \vspace{2.5cm}
author:
  - Stian Jørstad Sulebak
  - Mikkel Svartveit
bibliography: "bibliography.bib"

# Formatting options
documentclass: report
papersize: a4
geometry:
  - margin=2.5cm
fontsize: 10pt
mainfont: "SourceSerif4"
mainfontoptions:
  - Path=./assets/fonts/
  - Extension=.ttf
  - UprightFont=*-Regular
  - BoldFont=*-SemiBold
  - ItalicFont=*-Italic
  - BoldItalicFont=*-SemiBoldItalic
linestretch: 1.25
numbersections: true
link-citations: true
linkcolor: "RoyalBlue"
urlcolor: "RoyalBlue"
citecolor: "RoyalBlue"
filecolor: "RoyalBlue"

header-includes:
  - \usepackage{array}
  - \usepackage[all]{nowidow}
  - \usepackage[nohyperlinks]{acronym}
  - \usepackage{dirtree}
  - \usepackage{xcolor}
  - \usepackage[section]{placeins}
  - \usepackage[font={small},labelfont={bf}, width=0.8\textwidth]{caption}
  - \usepackage[norwegian,english]{babel}
  - \renewcommand{\maketitle}{} # Disable cover page
  - \usepackage{listings}
---

<!-- Abstract in English -->

\begin{abstract}
This thesis investigates the application of \acp{CNN} for detecting instruction set features from the binary code of a computer program, without explicit feature engineering. While prior research have focused on classifying the entire \ac{ISA} of a binary file, we shift focus to detecting individual architectural properties that can generalize across unknown or undocumented instruction sets. We train and evaluate six \ac{CNN} architectures of varying complexity on detecting endianness and fixed/variable instruction width, comparing simple models against deeper networks, one-dimensional versus two-dimensional convolutions, and models with and without embedding layers.

Using rigorous evaluation methods including \ac{LOGO CV} and cross-dataset testing, we demonstrate that small \ac{CNN} models with embedding layers can detect endianness with up to 90.3\% accuracy and fixed/variable instruction width with up to 88.0\% accuracy on unseen architectures within the ISAdetect dataset. However, we observe significant performance degradation when evaluating on more diverse datasets, with accuracy for unseen architectures dropping below 75\% for both target features. This indicates generalization challenges when models encounter truly novel architectures.

We introduce BuildCross, a new dataset containing binary code from 40 different \acp{ISA} with associated cross-compilation framework. Combining BuildCross with ISAdetect for training improves fixed/variable instruction width detection on the CpuRec dataset from approximately 55\% to over 80\%, demonstrating that architectural diversity in training data is important for generalization.

Our results show that while \acp{CNN} can detect key \ac{ISA} features from raw binary code with performance comparable to methods using manual feature engineering, their generalization capabilities are heavily dependent on training data diversity. This research contributes to advancing software reverse engineering capabilities, particularly for analyzing binaries from unknown or undocumented instruction set architectures.
\end{abstract}

<!-- Abstract in Norwegian -->

\selectlanguage{norwegian}
\begin{abstract}
Denne oppgaven undersøker bruken av konvolusjonsnettverk (CNN) for å identifisere egenskaper ved instruksjonssettet til et dataprogram direkte fra binærkode, uten eksplisitt databearbeiding. Tidligere forskning har fokusert på å klassifisere hele instruksjonssettarkitekturen (ISA) til en binærfil. Vår tilnærming er i stedet å identifisere individuelle arkitekturegenskaper som kan generaliseres til ukjente eller udokumenterte instruksjonssett. Vi trener og evaluerer seks CNN-arkitekturer med varierende kompleksitet for å oppdage endianness (byterekkefølge) og fast/variabel instruksjonsbredde. Vi sammenligner enkle modeller med dypere nettverk, endimensjonale mot todimensjonale konvolusjoner, samt modeller med og uten embedding-lag.

Ved bruk av omfattende evalueringsmetoder som kryssvalidering med usett gruppe samt testing på tvers av datasett, viser vi at små CNN-modeller med embedding-lag kan oppdage endianness med opptil 90,3 \% nøyaktighet og fast/variabel instruksjonsbredde med opptil 88,0 \% nøyaktighet på usette arkitekturer i ISAdetect-datasettet. Vi observerer imidlertid en betydelig reduksjon i ytelse ved evaluering på mer varierte datasett, der nøyaktigheten for usette arkitekturer faller under 75 \% for begge egenskapene. Dette tyder på generaliseringsutfordringer når modellene møter helt ukjente arkitekturer.

Vi introduserer BuildCross, et nytt datasett som inneholder binærkode fra 40 forskjellige \acp{ISA} med tilhørende rammeverk for krysskompilering. Ved å kombinere BuildCross med ISAdetect for trening forbedres deteksjonen av fast/variabel instruksjonsbredde på CpuRec-datasettet fra omtrent 55 \% til over 80 \%, noe som viser at arkitekturmangfold i treningsdataen er kritisk for generalisering.

Våre resultater viser at selv om CNNer kan oppdage sentrale ISA-egenskaper fra binærkode på nivå med metoder som benytter manuell databearbeiding, avhenger modellenes generaliseringsevne av mangfoldet i treningsdataen. Denne forskningen bidrar til å fremme ''reverse engineering'' av programvare, spesielt for analyse av binærfiler fra ukjente eller udokumenterte instruksjonssettarkitekturer.
\end{abstract}
\selectlanguage{english}

<!-- Override autoref behavior -->

\def\chapterautorefname{Chapter}
\def\sectionautorefname{Section}
\def\subsectionautorefname{Section}
\def\subsubsectionautorefname{Section}
\def\paragraphautorefname{Section}

<!-- Table of contents -->

```{=latex}
{
  \hypersetup{hidelinks}
  \tableofcontents
}
```

# Acronyms {-}

```{=latex}
\begin{acronym}
\acro{CISC}[CISC]{Complex Instruction Set Computing}
\acro{CNN}[CNN]{Convolutional Neural Network}
\acro{ELF}[ELF]{Executable and Linkable Format}
\acro{GCC}[GCC]{GNU Compiler Collection}
\acro{GPL}[GPL]{GNU General Public License}
\acro{GPU}[GPU]{Graphics Processing Unit}
\acro{HPC}[HPC]{High Performance Computing}
\acro{IoT}[IoT]{Internet of Things}
\acro{ISA}[ISA]{Instruction Set Architecture}
\acro{LOGO CV}[LOGO CV]{Leave-One-Group-Out Cross-Validation}
\acro{MMCC}[MMCC]{Microsoft Malware Classification Challenge}
\acro{NTNU}[NTNU]{the Norwegian University of Science and Technology}
\acro{RISC}[RISC]{Reduced Instruction Set Computing}
\acro{p.p.}[p.p.]{percentage points}
\acro{SGD}[SGD]{United Nations Sustainable Development Goals}
\end{acronym}
```
