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
  - Stian J. Sulebak
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
---

<!-- Abstract in English -->

\begin{abstract}
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt labore. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat in varius, temporibus et semper, facilisi. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur, sed ut perspiciatis unde omnis iste natus error similique et maxime. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum, while ac sapien inceptos himenaeos justo etiam phasellus turpis nullam fringilla semper, rhoncus sem, vitae. Integer nec odio praesent libero sed cursus ante dapibus diam, maecenas faucibus mollis interdum efficitur justo in tellus tempor, nunc. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis. At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi rerum facilis est cumque, optime. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et doloremque magnam aliquam.
\end{abstract}

<!-- Abstract in Norwegian -->

\selectlanguage{norwegian}
\begin{abstract}
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt labore. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat in varius, temporibus et semper, facilisi. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur, sed ut perspiciatis unde omnis iste natus error similique et maxime. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum, while ac sapien inceptos himenaeos justo etiam phasellus turpis nullam fringilla semper, rhoncus sem, vitae. Integer nec odio praesent libero sed cursus ante dapibus diam, maecenas faucibus mollis interdum efficitur justo in tellus tempor, nunc. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis. At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi rerum facilis est cumque, optime. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et doloremque magnam aliquam.
\end{abstract}
\selectlanguage{english}

<!-- Override autoref behavior -->

\def\chapterautorefname{Chapter}
\def\sectionautorefname{Section}
\def\subsectionautorefname{Section}
\def\subsubsectionautorefname{Section}

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
