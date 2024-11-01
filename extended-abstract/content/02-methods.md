# Methods

The review methodology followed a three-step protocol: research identification, primary study selection, and data extraction. Using Scopus as our primary database, we constructed a search query combining keywords related CNN applications for binary code analysis:

> TITLE-ABS-KEY(( "CNN" OR "convolutional neural network" ) AND ( "object code" OR "machine code" OR "binary file" OR "binary program" ))

This query yielded 86 results, which were then filtered based on these inclusion criteria (IC):

1. The research applies CNN to binary code of some sort.
2. The method does not require reverse engineering of the binary prior to analysis.
3. The method is not overly specialized for a specific task that would prevent its transfer to other target features.

Table \ref{table:paper-exclusion-results} shows the remaining papers after each step of the filtering process.

| IC screening step  | Articles reviewed | Articles excluded |
| ------------------ | ----------------: | ----------------: |
| Abstract           |                86 |                46 |
| Full-text          |                40 |                20 |
| Included in review |            **20** |                 – |

Table: Articles remaining after applying inclusion criteria. \label{table:paper-exclusion-results}

The remaining 20 papers were then subjected to a systematic review of their methodology, results, and limitations. We read through each article in its entirety, and labeled each study with the following primary categories:

- Application and targeted features
- Dataset used
- Binary encoding method
- Transfer learning approach (if any)
- CNN architecture
