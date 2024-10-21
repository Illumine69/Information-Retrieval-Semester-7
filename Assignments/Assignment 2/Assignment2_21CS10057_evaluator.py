import sys
import math

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Error! Usage: python Assignment2_21CS10057_evaluator.py <path_to_gold_standard_ranked_list.txt> <path_to_Assignment2_21CS10057_ranked_list_<K>.txt>")
        sys.exit(1)

    gold_standard_path = sys.argv[1]
    ranked_list_path = sys.argv[2]

    # Get ranked documents for each query
    print(f"Reading ranked list from {ranked_list_path}")
    ranked_list = {}
    with open(ranked_list_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            query_id, doc_ids = line.split(' : ')
            ranked_list[query_id] = doc_ids.strip().split()

    # Read the gold standard file and store relevance judgments
    print(f"Reading gold standard from {gold_standard_path}")
    relevance_scores = {}
    with open(gold_standard_path, 'r') as file:
        for line in file:
            query_id, _, doc_id, score = line.split()
            if query_id not in relevance_scores:
                relevance_scores[query_id] = {}
            relevance_scores[query_id][doc_id] = int(score)

    ap_10_dict = {}
    ap_20_dict = {}
    ndcg_10_dict = {}
    ndcg_20_dict = {}

    for query_id in ranked_list:
        print(f"Calculating metrics for query {query_id}")

        # Calculate AP@10 and AP@20
        ap_10 = 0
        ap_20 = 0
        num_relevant_docs = 0

        for i,doc_id in enumerate(ranked_list[query_id][:20]):
            if (doc_id in relevance_scores[query_id].keys()) and (relevance_scores[query_id][doc_id] > 0):
                num_relevant_docs += 1
                
            if i < 10:
                ap_10 += num_relevant_docs/(i+1)

            ap_20 += num_relevant_docs/(i+1)

        ap_10 /= 10
        ap_20 /= 20

        ap_10_dict[query_id] = round(ap_10,5)
        ap_20_dict[query_id] = round(ap_20,5)

        # Calculate NDCG@10 and NDCG@20
        ndcg_10 = 0
        ndcg_20 = 0
        
        for i,doc_id in enumerate(ranked_list[query_id][:20]):
            if doc_id in relevance_scores[query_id].keys() and relevance_scores[query_id][doc_id] > 0:
                if i < 10:
                    ndcg_10 += (math.pow(2,relevance_scores[query_id][doc_id])-1)/math.log2(i+2)

                ndcg_20 += (math.pow(2,relevance_scores[query_id][doc_id])-1)/math.log2(i+2)

        # Calculate ideal DCG@10 and DCG@20
        ideal_dcg_10 = 0
        ideal_dcg_20 = 0

        score_list = sorted(relevance_scores[query_id].values(), reverse=True)[:20]
        
        for i,score in enumerate(score_list):
            if i < 10:
                ideal_dcg_10 += (math.pow(2,score)-1)/math.log2(i+2)

            ideal_dcg_20 += (math.pow(2,score)-1)/math.log2(i+2)

        if ideal_dcg_10 == 0:   # Handle case when no relevant documents are present in gold standard for a query
            ndcg_10 = 0
        else:
            ndcg_10 /= ideal_dcg_10
            
        if ideal_dcg_20 == 0:
            ndcg_20 = 0
        else:
         ndcg_20 /= ideal_dcg_20

        ndcg_10_dict[query_id] = round(ndcg_10,5)
        ndcg_20_dict[query_id] = round(ndcg_20,5)

    mAP_10 = round(sum(ap_10_dict.values())/len(ap_10_dict),5)
    mAP_20 = round(sum(ap_20_dict.values())/len(ap_20_dict),5)
    mNDCG_10 = round(sum(ndcg_10_dict.values())/len(ndcg_10_dict),5)
    mNDCG_20 = round(sum(ndcg_20_dict.values())/len(ndcg_20_dict),5)

    K = ranked_list_path.split('_')[-1].split('.')[0]

    with open(f"Assignment2_21CS10057_metrics_{K}.txt", 'w') as file:
        file.write(f"{'Query_id' : <10}{'AP@10' : ^15}{'AP@20' : ^15}{'NDCG@10' : ^15}{'NDCG@20' : ^15}\n")
        for query_id in ranked_list:
            file.write(f"{query_id : <10}{ap_10_dict[query_id] : ^15}{ap_20_dict[query_id] : ^15}{ndcg_10_dict[query_id] : ^15}{ndcg_20_dict[query_id] : ^15}\n")

        file.write(f"\nMean Average Precision@10: {mAP_10}\nMean Average Precision@20: {mAP_20}\nMean Normalized Discounted Cumulative Gain@10: {mNDCG_10}\nMean Normalized Discounted Cumulative Gain@20: {mNDCG_20}\n")







        
