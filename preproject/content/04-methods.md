# Methodology

This section defines the review protocol used for this structured literature review, and provides justification for the strategies used in the context of our research questions. We outline our methodology for identifying and selecting relevant studies, including inclusion criteria, quality assessment standards for primary sources, and the data extraction process. The overall review protocol is identical for the two parts of the literature review. Differences in inclusion criteria and data extraction are explained and justified. The review protocol is split into three phases:

1. Research identification through database selection and querying
2. Filtration and selection of primary studies based on inclusion criteria
3. Data extraction and synthesis

## Database and search queries

Multiple research databases were considered, with our primary concern being including papers from peer-reviewed sources. Our research database of choice is Scopus [@Scopus], which from our experience includes results from a comprehensive list of peer-reviewed journals, workshops, and conferences. We acknowledge that the lack of diversity in research databases may result in missing relevant research. However, preliminary searches in IEEE and ACM revealed that all relevant results were already captured in Scopus, producing mostly duplicate entries. We deem Scopus as sufficient for the purpose of this review, considering the similar results from querying IEEE and ACM directly as well as the ease of consistent data extraction.

Using the "Advanced Search" feature in Scopus, we combine relevant keywords from both of our topics into two separate search queries, one for ML-ISA and one for CNN-part of this review. These search terms were grouped based on similarity and combined in conjunctive normal form. The final queries and result counts can be seen in Table \ref{table:search-queries}.

<!-- TODO: Inital preliminary research n ML-ISA provided by our supervisor -->

\begin{table}
\caption{Scopus search queries for each topic and result count}
\label{table:search-queries}\tabularnewline
\centering
\begin{tabular}{ |c|m{0.7\textwidth}|c| }
\hline
Topic & Query & Results \\
\hline
ML-ISA
& TITLE-ABS-KEY(( "machine learning" OR "deep learning" OR "neural network" ) AND \ ( "binary files" OR "binary code" OR "object code" OR "machine code" OR "binary program" ) AND \ ( "ISA" OR "instruction set" OR "target architecture" OR "reverse engineering" ))
& 74 \\
\hline
CNN
& TITLE-ABS-KEY(( "CNN" OR "convolutional neural network" ) AND ( "object code" OR "machine code" OR "binary file" OR "binary program" ))
& 86 \\
\hline
\end{tabular}
\end{table}

## Inclusion criteria and quality assessment

All 162 identified research papers across ML-ISA and CNN were assessed using inclusion and quality criteria, and excluded from the review if the paper is not deemed relevant. This leaves us with a selection of high quality and highly relevant primary studies to be used in the review. The inclusion criteria (IC) and quality assessment (QA) were selected based on our research questions for each topic, and applied through three steps:

- Exclusion of papers after applying IC on the abstract
- Exclusion of papers after applying IC on full-text
- Exclusion of papers after QA on full-text

ML-ISA inclusion criteria focuses on filtering out papers that does not apply machine learning to binary code of unknown ISA, i.e. without disassembly. In addition, for the paper to be relevant in the field, we require that the proposed method attempts to aid software reverse engineering. CNN inclusion criteria are a bit less strict, and does not require the paper to focus on reverse engineering specifically. We want to discover ways of applying CNN's on binary code directly, which means mainly excluding papers that require disassembled input. The specific IC's and QA's for both topics are:

<!-- **(Some mention of quality assesment would be nice ? )** -->

ML-ISA:

1. The research attempts to reverse engineer or aid in reverse engineering of raw binaries.
2. The method involves application of machine learning techniques on binary programs.
3. The method does not require disassembly, i.e. full knowledge of ISA information for the binary.

CNN:

1. The research applies CNN to binary code of some sort.
2. The method does not require reverse engineering of the binary prior to analysis.
3. The method is not overly specialized for a specific task that would prevent its transfer to other target features.

| Screening step     | Articles reviewed | Articles excluded |
| ------------------ | ----------------: | ----------------: |
| IC (Abstract)      |                74 |                43 |
| IC (Full-text)     |                31 |                25 |
| Included in review |             **6** |                 – |

Table: Articles remaining after applying inclusion criteria for ML-ISA studies. \label{table:ml-isa-exclusion-results}

| Screening step     | Articles reviewed | Articles excluded |
| ------------------ | ----------------: | ----------------: |
| IC (Abstract)      |                86 |                46 |
| IC (Full-text)     |                40 |                20 |
| Included in review |            **20** |                 – |

Table: Articles remaining after applying inclusion criteria for CNN studies. \label{table:cnn-exclusion-results}

## Data extraction process

In this subsection we provide a short description of how the resulting 26 primary studies were processed and studied. Each paper was read through in its entirety in order to identify potential avenues of comparison. In addition, we used the large language model Claude 3.5 Sonnet in order to create a summary of the research and highlight key aspects of the strategies employed in each paper.

Based on this and our research questions, we labelled each paper based on categories relevant for answering our RQ's. The categories used for the ML-ISA part of the review was **machine learning architecture (RQ1.1)**, **notable feature engineering techniques (RQ1.2)**, **datasets (RQ1.3)**, and **type of ISA identification or main contributions (RQ1.3)**. Relevant information from each RQ was then gathered, and synthesized into the 3 topics presented in our result: Feature engineering and feature extraction, Machine learning architecture and ISA classification targets.

In the CNN part of the review, we used labels identifying **datasets (RQ2.1)**, **encoding strategies of binary file as input (RQ2.2)**, **CNN architecture design (RQ2.3)**, and **use of pre-trained models (RQ2.3)**. To compare different CNN architectures according to **RQ2.4** we grouped and listed performance metrics for papers that target the same feature given a comparatively similar targeted domain. We chose to gather and report accuracy, precision, recall and F1-score of all papers where a performance comparison made sense, i.e. given similar datasets and targeted features. The data was synthesized into the four topics: Applications, Encoding binary data, Transfer Learning and CNN variations.
