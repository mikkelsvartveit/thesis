# Methododlology

This section defines the review protocol used for this structured literature review, and provides justification for the strategies used with regards to answering our research questions. This outlines methods for identification of relevant studies to be considered included our review, describing selected inclusion crieria and quality assessment for primary sources and the data extraction process for acquiering results. The overall review protocol is identical for literature regarding ML for ISA identification and CNN's applied to binary programs, where any differences in inclusion criteria and data extraction are explained. The review protocol is split into 3 phases: research identification through database selection and querying, filtering and selection of primary studies based on inclusion criteria, and data extraction and synthesis.

## Database and Search queries

Multiple research databases were considered, with our primary concern being including papers from peer-reviewed sources. Our research database of choice is Scopus (SOURCE?), as it from our experience includes results from a comprehensive list of peer reviewed journals, workshops and conferences. We recognize that the lack of diversity in research databases can result in missing out relevant work. However, preliminary work quering databases like IEEE and ACM compared to Scopus in our experiences only included duplicate results. We deem Scopus as sufficient for the purpose of this review, given similar results from quering IEEE and ACM directly and ease of consistent data extraction.

Using the "Advanced Search" feature in Scopus, we combined relevant keywords from both of our topics into two seperate search queries, one for ML-ISA and one for CNN-part of this review. These search terms were grouped based on similarity and combined in conjunctive normal form. The final queries and result counts can be seen in Table \ref{table:search-queries}.

\begin{table}[h!]
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
\caption{Scopus search queries for each topic and result count}
\label{table:search-queries}
\end{table}

## Inclusion criteria and quality assessment

All 160 identified research papers across ML-ISA and CNN were assessed using inclusion and quality criteria, and excluded from the review if the paper is not deemed relevant. This leaves us with a selection of high quality and highly relevant primary studes to be used in the review. The inclusion criteria (IC) and quality assesment (QA) were selected based on our research questions for each topic, and applied through three steps:

- Exclusion of papers after applying IC on the abstract
- Exclusion of papers after applying IC on full-text
- Exclusion of papers after QA on full-text

ML-ISA inclusion critera focuses on filtering out papers that does not apply machine learning to binary code of unknown ISA, i.e. without disassembly. In addition for the paper to be relevant in the field, we require that the proposed method attempts to aid software reverse engineering. CNN inclusion criteria are a bit less strict, and does not require the paper to focus on reverse engineering specifically. We want to discover ways of applying CNN's on binary code directly, which means mainly excluding papers that require disassembled input. 

(Some mention of quality assesment would be nice ? )


\begin{table}[h!]
\centering
\begin{tabular}{ |c|c|c|c| }
\hline
Topic & DB query result & IC on abstract & IC on full-text \\
\hline
ML-ISA
& 74 
& 31 (-43)
& 8 (-23) \\
\hline
CNN
& 86 
& 41 (-45)
& 22 (-19) \\
\hline
\end{tabular}
\caption{Papers left after applying inclusion criteria. tabellen ble litt shit, finne hvordan visualisere bedre}
\label{table:paper-exclusion-results}
\end{table}

## Data extraction process
