#!/usr/bin/python3.9

import ranx
import pprint
from collections import OrderedDict,defaultdict
from pprint import pprint
from ranx import  Run,fuse,Qrels
from ranx.fusion import probfuse,probfuse_train

# check fusion methods are in the package:
#print(dir(ranx.fusion))

# create ordered dictionaries from files
def load_cleaned_run_ordered(filepath):
    run = OrderedDict()
    with open(filepath, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            qid, docid, score = line.strip().split()
            q_key = f"q_{qid}"
            d_key = f"d_{docid}"
            score = float(score)
            if q_key not in run:
                run[q_key] = OrderedDict()
            run[q_key][d_key] = score
    return run

# load and create dictionary for qrels from file
def load_qrels_ordered_with_ranks(filepath, max_rel_score=3):
    qrels = defaultdict(OrderedDict)
    current_qid = None
    rel_score = max_rel_score

    with open(filepath, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            qid, docid = line.strip().split()
            q_key = f"q_{qid}"
            d_key = f"d_{docid}"

            # If the query changes, begin from the max_rel_score
            if current_qid != qid:
                current_qid = qid
                rel_score = max_rel_score

            # Only if document is not added
            if d_key not in qrels[q_key]:
                qrels[q_key][d_key] = rel_score
                rel_score = max(1, rel_score - 1) # goes down until 1

    return qrels

# load the files and create dictionaries for each similarity method
bm25 = load_cleaned_run_ordered("bm25_cleaned")
LMDirichletSimilarity = load_cleaned_run_ordered("LMDirichletSimilarity_cleaned")
LMJelinekMercerSimilarity = load_cleaned_run_ordered("LMJelinekMercerSimilarity_cleaned")
tf_idf = load_cleaned_run_ordered("tf_idf_cleaned")
# check the dictionaries
# pprint.pprint(bm25)

# load qrels with relevance documents
qrels = load_qrels_ordered_with_ranks("qrels.text_parsed_2_cleaned")
#pprint(qrels)

# Train the Model
run_bm25 = Run(bm25)
run_LMDirichletSimilarity = Run(LMDirichletSimilarity)
run_LMJelinekMercerSimilarity = Run(LMJelinekMercerSimilarity)
run_tf_idf = Run(tf_idf)
run_qrels = Qrels(qrels)

# for small data sets is good 2 , for large is 4
n_segments = 2

# Train the model and compute the probabilities
probs = probfuse_train(run_qrels, [run_bm25, run_LMDirichletSimilarity, run_LMJelinekMercerSimilarity, run_tf_idf], n_segments)

# The dictionaries for the similarity methods : bm25, LMDirichletSimilarity, LMJelinekMercerSimilarity, tf_idf
# These dictionaries fill for query 1: What articles exist which deal with TSS (Time Sharing System), an operating system for IBM computers?
# with top 10 output results from the above methods from Lucene
# Then we create the top 10 output documents based on the previous probabilities of the other 51 trained results.
# No need here qrels for that

bm25_1 = { 
    "q_1": {
        "d_1938": 15.568998,
        "d_2629": 11.773457,
        "d_1752": 10.79806,
        "d_1519": 10.79806,
        "d_1657": 9.924984,
        "d_1827": 9.924984,
        "d_3127": 9.32327,
        "d_2219": 9.202567,
        "d_1544": 9.202567,
        "d_1523": 9.202567
    }   
}

LMDirichletSimilarity_1 = {
    "q_1": {
        "d_2319": 3.7196512,
        "d_1410": 3.6318643,
        "d_397": 2.9093661,
        "d_1046": 2.7099025,
        "d_2340": 1.9826454,
        "d_1591": 1.9049209,
        "d_1680": 1.8736409,
        "d_2317": 1.8630118,
        "d_3069": 1.8361032,
        "d_1572": 1.5588756,
    }
}

LMJelinekMercerSimilarity_1 = {
    "q_1": {
        "d_1938": 24.936703,
        "d_1657": 24.655859,
        "d_2629": 19.74606,
        "d_1752": 19.744024,
        "d_1519": 19.744024,
        "d_1827": 16.860128,
        "d_3127": 16.120739,
        "d_2219": 15.85324,
        "d_1544": 15.85324,
        "d_1523": 15.85324,
    }
}

tf_idf_1 = {
    "q_1": {
        "d_2319": 15.872329,
        "d_2629": 15.701583,
        "d_1938": 15.468782,
        "d_1657": 15.272469,
        "d_1752": 12.719387,
        "d_1519": 12.719387,
        "d_1827": 12.162394,
        "d_3127": 11.214344,
        "d_1544": 11.102695,
        "d_1523": 11.102695,
    }
}

run_1 = Run(bm25_1)
run_2 = Run(LMDirichletSimilarity_1)
run_3 = Run(LMJelinekMercerSimilarity_1)
run_4 = Run(tf_idf_1)

# the propabilistic method probfuse does not need score normalization because cares only for ranking and produces propabilities based on document rankings.
combined_run = probfuse([run_1, run_2, run_3, run_4], probs)
# print (combined_run)


top_10 = list(combined_run["q_1"].items())[:10]
for rank, (doc, score) in enumerate(top_10, start=1):
    doc_cacm = doc.replace("d_", "cacm")
    print(f"Rank{rank}: {doc_cacm} with score {round(score, 3)}")


