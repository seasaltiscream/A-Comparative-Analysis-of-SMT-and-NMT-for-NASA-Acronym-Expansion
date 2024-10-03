# A-Comparative-Analysis-of-SMT-and-NMT-for-NASA-Acronym-Expansion
This project explores the effectiveness of Neural Machine Translation (NMT) and Statistical Machine Translation (SMT) in expanding acronyms, specifically focusing on those used by NASA in technical documentation. Using a unique dataset of 484 acronyms and their definitions provided by NASA, we assess the accuracy of both translation algorithms. 

**File details**
- processed_acronyms.jsonl:
Each line in this file is an acronym found to have more than one defintion. There are 484 acronyms found with multiple definitions suitable for model building. Each line contains information on acronym, definitions, and where found in the corpus. The corpus is the file results_merged.jsonl

- smt4.ipynb:
This file loads a dataset of acronyms and their definitions from a JSONL file, generates acronyms from definitions using tokenization, and evaluates the match rate between generated and actual acronyms through a test set.

- nmt4.ipynb:
This file loads a dataset of acronyms and definitions from a JSONL file, preprocesses the data using a tokenizer, and defines a Neural Machine Translation (NMT) model utilizing LSTM layers to expand acronyms. It also includes training and evaluation steps to assess the model's performance on a test set.
