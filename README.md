# Towards Broader Language Representation in IR: Challenges from Arabic IR and a Path Forward

_A perspective paper submitted for review to SIGIR-AP 2026_

This repository accompanies a **Perspective Track** submission to **SIGIR-AP 2026**, held in Singapore.  
It provides the data required to reproduce the reported results, as well as a reference repository containing an up-to-date list of Arabic IR research used in the paper.

---

## Abstract
Information Retrieval (IR) research has made remarkable progress over the past decades, shaping how people access information worldwide. As global information access continues to expand, there is an increasing opportunity to better represent the diversity of languages, cultures, contexts, and information needs. In this per-
spective paper, we examine how core IR components, including evaluation metrics, user models, and system design assumptions, have primarily evolved from English-language test collections and user studies. Analyzing research from leading IR venues over the past decade, we identify opportunities to broaden non-English rep-
resentation. Using Arabic as a case study, we highlight key linguistic challenges and present an empirical analysis of dialectal variation, demonstrating its impact on retrieval effectiveness and stability.
We conclude with a community-driven roadmap to advance Arabic IR and practical recommendations to foster broader inclusion of diverse languages in future IR research

---

## Repository Structure

```
├── Arabic-IR-papers.md #An up-to-date list and metadata of reviewed Arabic IR papers.
├── language-representation/ #Bibliometric data and the prompt used to verify language mentions (Section 2).
│ ├── raw-bib-files/  #Raw bibliometric data of included papers
├── dialectal-query-variants/ #Collected dialectal query variants, code, and generated runs (Section 5)
│ ├── query_variants/  # dialectal query variants
│ ├── code/ # experimental code for running etrieval models and evaluating the results.
│ ├── runs_dialect/ # runs generated from running retrieval models on all dialects
└── README.md
```
