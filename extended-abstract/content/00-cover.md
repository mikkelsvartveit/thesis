---
title: "[Title undecided]"
subtitle: |
  Extended Abstract\
  \
  \small
  Department of Computer Science\
  Norwegian University of Science and Technology\
  Trondheim, Norway
date: \today
author:
  - Stian Sulebak
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
urlcolor: "blue"

abstract: |
  The increasing diversity of binary programs across embedded systems, IoT devices, and various digital platforms has created a need for effective binary code analysis techniques. We present a structured literature review examining the application of convolutional neural networks (CNN) to binary code analysis, with a particular focus on approaches that operate directly on raw binaries without requiring disassembly. Through a systematic review of 20 primary studies from the Scopus database, we analyze and compare different approaches based on their binary code representation methods, network architectures, and targeted features. Our findings reveal that while malware classification dominates the current applications (18 out of 20 studies), CNNs have also shown promise in detection of compiler optimization levels. 

  The review identifies several key trends: the prevalence of image-based binary representations, the effectiveness of transfer learning using pre-trained models like VGG-16, and the emergence of specialized CNN architectures designed specifically for binary analysis. We find that state-of-the-art approaches achieve accuracies above 99% in malware classification tasks, with recent innovations in network architecture and data preprocessing contributing to these improvements. This comprehensive analysis provides insights into the current state of CNN-based binary code analysis and identifies directions for future research in the field.

header-includes:
  - \usepackage{array}
---
