# Methodology

This section defines the review protocol used for this systematic literature review, and explains how the chosen methods align with our research questions. We outline our methodology for identifying and selecting relevant studies, including inclusion criteria, quality assessment standards for primary sources, and the data extraction process. The overall review protocol is identical for the two parts of the literature review. Differences in inclusion criteria and data extraction are explained and justified. The review protocol is split into three phases:

1. Research identification through database selection and querying
2. Filtration and selection of primary studies based on inclusion criteria and quality assessment
3. Data extraction and synthesis

## Database and search queries

Multiple research databases were considered, with our primary concern being including papers from peer-reviewed sources. Our research database of choice is Scopus [@Scopus], which from our experience includes results from a comprehensive list of peer-reviewed journals, workshops, and conferences. We acknowledge that the lack of diversity in research databases may result in missing relevant research. However, preliminary searches in IEEE and ACM revealed that all relevant results were already captured in Scopus, producing mostly duplicate entries. We deem Scopus as sufficient for the purpose of this review, considering the similar results from querying IEEE and ACM directly as well as the ease of consistent data extraction.

Using the "Advanced Search" feature in Scopus, we combine relevant keywords from both of our topics into two separate search queries, one of them covering machine learning for ISA detection (hereby referred to as ML-ISA) and the other covering CNN applied to binary code analysis (hereby referred to as CNN-BCA). The search terms are grouped based on similarity and combined in conjunctive normal form. The full queries and the number of results for each are shown in Table \ref{table:search-queries}.

This review was in part commissioned by our supervisor Donn Morrison in preparation for our master’s thesis. We were provided a list of relevant papers as part of the preliminary search. The purpose of this was to gain an overview of the field of machine learning aided reverse engineering. While we consider our search queries for the ML-ISA part of the review to encapsulate our goals in **RQ1**, some notable papers from the preparation phase did not show up in our search results. The papers in question are ELISA [@Nicolao2018] and Ma et al. [@Ma2019]. These articles are available on Scopus, but the chosen search terms did not match the in the title, abstract, or keywords of these studies. We have decided to include these papers in our review due to:

- Their high relevancy for our research questions
- Their presence on Scopus, which we consider a reputable scientific database
- The possibility that our knowledge of existing research might influence further analysis if not included in the review


\begin{table}
\def\arraystretch{2}
\caption{Scopus search queries for each topic and result count}
\label{table:search-queries}\tabularnewline
\vspace{0.5cm}
\centering
\begin{tabular}{ m{0.15\textwidth} m{0.7\textwidth} c }
\hline
Topic & Query & Results \\
\hline
ML-ISA
& TITLE-ABS-KEY(( "machine learning" OR "deep learning" OR "neural network" ) AND \ ( "binary files" OR "binary code" OR "object code" OR "machine code" OR "binary program" ) AND \ ( "ISA" OR "instruction set" OR "target architecture" OR "reverse engineering" ))
& 74 \\
Preliminary ML-ISA search
& Supplied by our supervisor, Donn Morrison
& 2 \\
CNN-BCA
& TITLE-ABS-KEY(( "CNN" OR "convolutional neural network" ) AND ( "object code" OR "machine code" OR "binary file" OR "binary program" ))
& 86 \\
\hline
\end{tabular}
\end{table}

## Inclusion criteria and quality assessment

We assessed all 162 identified research papers using inclusion criteria and quality assessment, excluding those found to be irrelevant to our two-part review. This leaves us with a selection of high quality and highly relevant primary studies to be further analyzed in the review. The inclusion criteria were selected based on our research questions for each topic. Articles were filtered through three steps:

- Exclusion of papers after applying inclusion criteria on the abstract
- Exclusion of papers after applying inclusion criteria on full text
- Exclusion of papers after quality assessment on full text

The ML-ISA inclusion criteria focuses on filtering out articles that do not require disassembling the binary prior to applying machine learning, as this requires detailed knowledge about the instruction set. In addition, for the article to be considered relevant, we require that the proposed method attempts to aid software reverse engineering. CNN-BCA inclusion criteria are less strict, and do not require the article to focus on reverse engineering specifically. We wish to discover ways of applying CNN to binary code directly, which predominantly implies excluding papers that require disassembled input. The specific inclusion criteria for both topics are:

ML-ISA:

1. The research attempts to reverse engineer or aid in reverse engineering of raw binaries.
2. The method involves application of machine learning techniques on binary programs.
3. The method does not require disassembly, i.e. full knowledge of ISA information for the binary.

CNN-BCA:

1. The research applies CNN to binary code of some sort.
2. The method does not require reverse engineering of the binary prior to analysis.
3. The method is not overly specialized for a specific task to the point where it would prevent transferability to other target features.

As for quality assessment, we focus on evaluating the reporting quality of the included studies using two primary criteria. We examine whether there is a clear statement of research goals, which ensures that objectives and scope of the article is clearly defined. We also want included papers to be contextualized within existing research, that it is built upon existing knowledge and contributes to the field. The specific quality assessment criteria are:

1. Is there is a clear statement of the aim of the research?
2. Is the study is put into context of other studies and research?

| Screening step     | Articles reviewed | Articles excluded |
| ------------------ | ----------------: | ----------------: |
| Abstract           |                76 |                43 |
| Full text          |                33 |                25 |
| Included in review |             **6** |                 – |

Table: Articles remaining after applying inclusion criteria and quality assessment for ML-ISA primary studies. \label{table:ml-isa-exclusion-results}

| Screening step     | Articles reviewed | Articles excluded |
| ------------------ | ----------------: | ----------------: |
| Abstract           |                86 |                46 |
| Full text          |                40 |                20 |
| Included in review |            **20** |                 – |

Table: Articles remaining after applying inclusion criteria and quality assessment for CNN-BCA primary studies. \label{table:cnn-exclusion-results}

## Data extraction process

In this subsection, we provide a short description of how the remaining 26 primary studies were processed and studied. Each article was read through in its entirety in order to identify potential avenues of comparison. In addition, we used the large language model Claude 3.5 Sonnet for extracting performance numbers and highlighting key aspects of the strategies employed in each article.

Based on this and our research questions, we label each paper based on categories relevant for answering our research questions. The categories used for the ML-ISA part of the review are machine learning architecture (**RQ1.1**), notable feature engineering techniques (**RQ1.2**), datasets (**RQ1.3**), and type of ISA identification or main contributions (**RQ1.3**). Relevant information for each research question is gathered and synthesized into the three topics presented in our results: Feature engineering & extraction, machine learning architecture, and ISA classification targets.

In the CNN-BCA part of the review, we label the articles with datasets (**RQ2.1**), encoding strategies of the binary files (**RQ2.2**), CNN architecture designs (**RQ2.3**), and use of pre-trained models (**RQ2.3**). To compare the performance of the different approaches (**RQ2.4**), we group and list performance metrics for studies in similar domains that target the same feature. We choose to gather and report accuracy, precision, recall, and F1-scores of all studies where a performance comparison makes sense, i.e. given similar datasets and target features. Qualitative data is synthesized into four topics: applications, encoding of binary data, transfer learning, and CNN variations.
