# Abstract

Automated test case generation reduces the manual
effort of software testing, particularly for dynamic languages
like Python. This study compares three transformer-based mod-
els—CodeT5, CodeBERT, and CodeGen. While CodeBERT and
CodeGen were expected to excel due to their focus on code under-
standing and generation, CodeT5, a code summarization model,
outperformed both, achieving higher code coverage and superior
semantic and syntactic fidelity. Using a limited yet diverse dataset
of Python functions with natural language descriptions and unit
tests, we evaluated models with NLP metrics (BLEU, ROUGE-L)
and practical measures like code coverage. CodeGen struggled
with coherence under data constraints, while CodeBERT showed
moderate effectiveness. Paired t-tests confirmed CodeT5’s sta-
tistically significant advantage, highlighting that a well-adapted
model can outperform more generalized architectures in realistic
software testing scenarios.
