---
title: "Machine Learning for Reverse Engineering & Convolutional Neural Networks for Binary Code Analysis: A Systematic Literature Review"
subtitle: |
  Specialization Project\
  \vspace{1cm}
  ![NTNU Logo](./images/ntnu.png){ width=12% }\
  \vspace{0.5cm}
  \small
  Department of Computer Science\
  Norwegian University of Science and Technology\
  Trondheim, Norway\
date: |
  \today
  \vspace{2.5cm}
author:
  - Stian J. Sulebak
  - Mikkel Svartveit
bibliography: "bibliography.bib"

# Formatting options
documentclass: extarticle
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
linkcolor: "blue"
urlcolor: "blue"
toc: "true"

abstract: |
  Binary reverse engineering is critical for analyzing security, quality, and compatibility of compiled programs. The increased demand of IoT devices leads to new challenges for reverse engineers, as embedded systems often use custom instruction set architectures (ISA). This systematic literature review examines two key areas in software reverse engineering: machine learning approaches for ISA detection and convolutional neural networks (CNN) for binary code analysis. Through a structured review of 26 primary studies, we analyze how machine learning techniques have been applied to classify ISA features and how CNN have been used for analyzing raw binary code. Our findings reveal that current machine learning approaches for ISA detection predominantly employ traditional models not based on deep learning. They achieve high accuracy in classifying known architectures, but face limitations in distinguishing similar architectures and handling non-code sections of the binary file. For CNN applications to binary code, we find strong evidence of effectiveness particularly in malware classification, with accuracies exceeding 99% on standard datasets without requiring manual feature engineering. However, CNN applications beyond malware detection remain limited. The review identifies significant research gaps, particularly in developing architecture-agnostic methods capable of identifying specific ISA features rather than classifying known architectures. We conclude that while current machine learning methods show promise, future research should focus on leveraging CNN's automatic feature learning capabilities while reducing reliance on binary format metadata.
include-before: |
  \pagebreak

header-includes:
  - \usepackage{array}
---

\pagebreak
