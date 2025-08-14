# Python
Ranx Python Library Fusion Methods Implementation

The script combsumrrfmax.py implements the methods for meta-search ranking : CombSUM, CombMAX, RRF

The script probfuse.py implements the method ProbFuse for meta-search ranking.

All the methods use the output results from Lucene similarities methods : BM25, TF-IDF, LMDirichletSimilarity, LMJelinekMercerSimilarity

The <cleaned> files used for train the model for ProbFuse method inside probfuse.py.As output is top-10 documents for query 1 
based on train dataset of 51 queries and their output results. 

The <top_50> files used as input scoring data and after perform fusion from file combsumrrfmax.py present top-10 documnets.

The qrels.text_parsed_2_cleaned_file, is the ground truth knowledge of judges (From CACM corpus) for the scored documents, used at probfuse.py.

For supported fusion algorithms and what is Ranx check : https://amenra.github.io/ranx/fusion/#supported-fusion-algorithms

For more information check the source code of both scripts.

All the above cobe is for implemente meta-search algorithms and better scoring evaluation.
