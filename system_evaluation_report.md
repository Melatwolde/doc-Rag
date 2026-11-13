# System Evaluation and Reliability Report

Answer accuracy is **grounded answer accuracy**: an answer is counted only when it does not abstain and its retrieved evidence contains the labelled relevant passage. Citation accuracy requires a citation to such a passage.

| Threshold | Grounded answer accuracy | Citation accuracy | Correct abstention | False abstention | Mean latency (ms) | P95 latency (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.20 | 61.5% | 57.1% | 66.7% | 0.0% | 17.01 | 18.93 |
| 0.45 | 61.5% | 61.5% | 100.0% | 0.0% | 13.47 | 15.35 |
| 0.70 | 46.2% | 60.0% | 100.0% | 23.1% | 16.33 | 19.19 |

The evaluation includes 13 labelled answerable retrieval queries and 3 explicitly unanswerable queries. It reports measured values only when this script is run.
