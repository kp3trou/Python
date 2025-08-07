#!/usr/bin/python3.9

import os
import sys
import ranx
from ranx import  Run,fuse,Qrels
from ranx.fusion import bordafuse,weighted_bordafuse,comb_sum,rrf
# check fusion methods are in the package:
#print(dir(ranx.fusion))

# returns dictionary {'q_1': {doc_id:score},{},...}
def load_file_run_dict(file_path):
    run_dict = {"q_1": {}}

    with open(file_path, "r") as file:
        for line in file:
            # Only process valid lines
            if "Query" in line and "Path" in line and "Combined Score" in line:
                parts = line.strip().split(",")
                if len(parts) != 2:
                    print(f"Skipping malformed line: {line.strip()}")
                    continue
                try:
                    # Extract doc ID
                    doc_path = parts[0].split("Path:")[1].strip()
                    doc_id = os.path.basename(doc_path)

                    # Extract score
                    score = float(parts[1].split("Combined Score:")[1].strip())

                    # Assign to dictionary
                    run_dict["q_1"][doc_id] = score
                except Exception as e:
                    print(f"Error parsing line: {line.strip()} -> {e}")
                    continue

    return run_dict

# Pass fusion method as argument from command line
if len(sys.argv) != 2:
    print("Usage: ./combsumrrfmax.py [combsum | combmax | rrf]")
    sys.exit(1)

fusion_method = sys.argv[1].lower()
valid_methods = ["combsum", "combmax", "rrf"]

if fusion_method not in valid_methods:
    print(f"Unknown Method : {fusion_method}")
    print("Accepted Values : combsum, combmax, rrf")
    sys.exit(1)


# files with top 50 results 
file_path_bm25 = "lucene_output_bm25_top_50_query_1"
file_path_tf_idf = "lucene_output_tf_idf_top_50_query_1"
file_path_LMJelinekMercerSimilarity = "lucene_output_LMJelinekMercerSimilarity_top_50_query_1"
file_path_LMDirichletSimilarity = "lucene_output_LMDirichletSimilarity_top_50_query_1"

# dictionary creation
run_lucene_output_bm25_top_50 = load_file_run_dict(file_path_bm25)
run_lucene_output_tf_idf_top_50 = load_file_run_dict(file_path_tf_idf)
run_lucene_utput_LMJelinekMercerSimilarity_top_50 = load_file_run_dict(file_path_LMJelinekMercerSimilarity)
run_lucene_output_LMDirichletSimilarity_top_50 = load_file_run_dict(file_path_LMDirichletSimilarity)

# Print the entire dictionary for debugging
#print(run_lucene_output_bm25_top_50)

# create Run objects
run_bm25 = Run(run_lucene_output_bm25_top_50)
run_tf_idf = Run(run_lucene_output_tf_idf_top_50)
run_LMJelinekMercerSimilarity = Run(run_lucene_utput_LMJelinekMercerSimilarity_top_50)
run_LMDirichletSimilarity = Run(run_lucene_output_LMDirichletSimilarity_top_50)
#print(run_bm25)

# Choose Fusion Method
if fusion_method == "combsum":
    combined_run = fuse(
        runs=[run_bm25, run_tf_idf, run_LMJelinekMercerSimilarity, run_LMDirichletSimilarity],
        norm="min-max", # Normalization strategy
        method="sum" # Alias for CombSUM
    )
elif fusion_method == "combmax":
    combined_run = fuse(
        runs=[run_bm25, run_tf_idf, run_LMJelinekMercerSimilarity, run_LMDirichletSimilarity],
        norm="min-max", # Normalization strategy
        method="max" # Alias for CombMAX
    )
elif fusion_method == "rrf":
    combined_run = fuse(
        runs=[run_bm25, run_tf_idf, run_LMJelinekMercerSimilarity, run_LMDirichletSimilarity],
        norm=None,
        method="rrf",
        params={"k": 60}
    )

#print (combined_run)

# print results
for qid in combined_run.keys():
    top_docs = sorted(combined_run[qid].items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top 10 docs for query {qid}:")
    for rank, (doc_id, score) in enumerate(top_docs, start=1):
        print(f"  Rank {rank}: {doc_id} with score {round(score, 3)}") # round at third decimal


