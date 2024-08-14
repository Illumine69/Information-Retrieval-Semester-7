import pickle
import sys

def merge_list(postings_list1, postings_list2):
    result = []
    i=0
    j=0

    while i < len(postings_list1) and j < len(postings_list2):
        id1 = postings_list1[i]
        id2 = postings_list2[j]

        if id1 == id2:
            result.append(id1)
            i += 1
            j += 1
        elif id1 < id2: # move forward as postings are sorted
            i += 1
        else:
            j += 1

    return result

if __name__ == "__main__":
    # Error check to ensure correct python command
    if len(sys.argv) != 3:
        print("Error! Usage: python Assignment1_21CS10057_bool.py <path to model> <path to query file>")
        sys.exit(1)

    model_path = sys.argv[1]
    query_path = sys.argv[2]

    inverted_index = pickle.load(open(model_path, 'rb'))
    with open(query_path, 'r') as query_file, open('Assignment1_21CS10057_results.txt', 'w') as results_file:
        for query in query_file:
            # get query id and corresponding tokens
            id, tokens = query.strip().split('\t')
            tokens = tokens.split()

            # get postings list for first token
            result = inverted_index.get(tokens[0],[])

            # perform AND of all the postings list
            for token in tokens[1:]:
                postings_list = inverted_index.get(token, [])
                result = merge_list(result, postings_list)

            # write the results
            results_file.write(f"{id} : {' '.join(map(str, result))}\n")

    print("Results written to Assignment1_21CS10057_results.txt file")
