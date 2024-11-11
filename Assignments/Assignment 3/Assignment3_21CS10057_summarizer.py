import nltk
from nltk.tokenize import sent_tokenize
import sys
import pandas as pd
import spacy
import string
import math
import glpk
import numpy as np
from nltk.stem import PorterStemmer

# Download the required nltk packages
nltk.download('punkt_tab')

# Set the maximum length of the summary
K = 200

# Create a stemmer object
stemmer = PorterStemmer()

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    '''
    Function: Remove stop words, punctuation marks and perform lemmatization (without POS tags)
              to generate tokens
    '''
    # Replace hyphens with spaces
    text = text.replace('-', ' ')

    # Lowercase the text and tokenize
    tokens = nlp(text.lower())

    # Remove stopwords and punctuations
    tokens = [token for token in tokens if token.text not in string.punctuation]    # Remove punctuations
    tokens = [token for token in tokens if token.text.isalpha()]                    # Remove numbers
    tokens = [token for token in tokens if not token.is_stop]                       # Remove stopwords
    tokens = [stemmer.stem(token.lemma_) for token in tokens]                       # Lemmatize and stem

    return tokens

def parseDataset(dataset_path):
    '''
    Function: Get id, articles and highlights from the csv file
    Input: Path to the dataset
    Output: Id and Articles
    '''
    articleDict = {}
    data = pd.read_csv(dataset_path)
    print("Parsing dataset...")
    for i in range(len(data)):
        print("Parsing article %d" % i)

        Docid = data['id'][i]
        article = data['article'][i]
        summary = data['highlights'][i]

        if len(preprocess_text(summary)) > K:
            print("Summary length exceeds K for article %s. Skipping..." % Docid)
            continue

        sentences = sent_tokenize(article)
        for sentence in sentences:
            tokens = preprocess_text(sentence)
            if Docid not in articleDict:
                articleDict[Docid] = []

            # Remove empty tokens
            if(len(tokens) > 0):
                articleDict[Docid].append(tokens)

    print("Dataset parsed!")
    return articleDict

def get_inverted_index(article_dict):
    '''
    Creates an inverted index with tokens as keys and article IDs as postings

    Returns a dictionary of token(keys) and id
    '''
    print("Creating inverted index...")
    inv_index_dict = {}
    for id, sentences in article_dict.items():
        for tokens in sentences:
            for token in tokens:
                if token not in inv_index_dict:
                    inv_index_dict[token] = {id: 1}
                else:
                    if id not in inv_index_dict[token]:
                        inv_index_dict[token][id] = 1
                    else:
                        inv_index_dict[token][id] += 1

    # Sort the postings list
    inv_index = {}
    for token in inv_index_dict:
        inv_index[token] = [(k, v) for k, v in sorted(inv_index_dict[token].items(), key=lambda item: item[0])]

    print("Inverted index created!")
    return inv_index

def tf_idf_docs(term_frequency, total_docs,total_terms):
    '''
    Calculate the TF-IDF score for each term-document pair using lnc scheme

    Returns the average document vector for the corpus
    '''

    print("Calculating TF-IDF scores for each term-document pair...")
    doc_tf_idf_vector = {}
    for doc in total_docs:
        doc_tf_idf_vector[doc] = {}

    # Iterate through the term frequency dictionary
    for (term, doc_id), tf in term_frequency.items():
        # Calculate the term frequency (TF) for the term-doc pair
        tf_idf = 1 + math.log(tf, 10) if tf > 0 else 0

        # Store the TF-IDF score in the dictionary
        doc_tf_idf_vector[doc_id][term] = tf_idf

    # Normalize the TF-IDF scores
    for doc_id in total_docs:
        doc_vector = doc_tf_idf_vector[doc_id]
        doc_norm = math.sqrt(sum([score ** 2 for score in doc_vector.values()]))
        doc_tf_idf_vector[doc_id] = {term: score / doc_norm for term, score in doc_vector.items()}

    # Average out the document vectors to get the corpus vector
    avg_doc_tf_idf_vector = {term: sum([doc_vector.get(term,0) for doc_vector in doc_tf_idf_vector.values()]) / len(total_docs) for term in total_terms}

    return avg_doc_tf_idf_vector

def tf_idf_sentences(sentence_term_frequency, document_frequency, total_docs, total_terms):
    '''
     Calculate the TF-IDF score for each term-sentence pair using the ltc scheme

     Returns a dictionary of TF-IDF vectors for each sentence in the corpus
    '''

    print("Solving TF-IDF scores for each term-sentence pair...")
    sentence_tf_idf_vector = {}
    for article_id, sentences in sentence_term_frequency.items():
        sentence_tf_idf_vector[article_id] = {}

        for i in sentences.keys():
            sentence_tf_idf_vector[article_id][i] = {}

    # Iterate through the query term frequency dictionary
    for article_id, sentences in sentence_term_frequency.items():
        for i, query_terms in sentences.items():
            for term, tf in query_terms.items():
                # Calculate the term frequency (TF) for the term-query pair
                tf = 1 + math.log(tf, 10) if tf > 0 else 0

                # Calculate the inverse document frequency (IDF) for the term
                idf = math.log(len(total_docs) / document_frequency[term], 10)

                # Calculate the TF-IDF score for the term-query pair
                tf_idf = tf * idf

                # Store the TF-IDF score in the dictionary
                sentence_tf_idf_vector[article_id][i][term] = tf_idf

            # Normalize the TF-IDF scores
            query_vector = sentence_tf_idf_vector[article_id][i]
            query_norm = math.sqrt(sum([score ** 2 for score in query_vector.values()]))
            sentence_tf_idf_vector[article_id][i] = {term: score / query_norm for term, score in query_vector.items()}

    return sentence_tf_idf_vector

def get_cosine_similarity(vector_1, vector_2):
    '''
    Function: Calculate the cosine similarity between two vectors
    '''

    # Ensure that vector_1 is the smaller vector
    if len(vector_1) > len(vector_2):
        vector_1, vector_2 = vector_2, vector_1

    dot_product = sum([vector_1[term] * vector_2.get(term, 0) for term in vector_1.keys()])
    return dot_product

def ILPFunction(doc_corpus_vector, doc_sentence_vector_dict, sentence_token_len, articleDict):
    '''
    Solve the ILP problem to get the most relevant sentences for summarization
    '''
    print("Solving ILP problem...")

    # Open the file to write the output
    f = open("Assignment3_21CS10057_summary.txt", "w")

    # Iterate through the articles
    for num, (article_id, sentences) in enumerate(doc_sentence_vector_dict.items()):
        print("Solving ILP problem for article %d" % num)
        lp = glpk.LPX()
        lp.name = 'sent' + str(article_id)
        lp.obj.maximize = True

        num_sentences = len(sentences)

        # Add the variables
        num_cols = (int)(num_sentences + num_sentences*(num_sentences - 1)/2)
        lp.cols.add(num_cols)

        for i in range(num_sentences):
            col = lp.cols[i]
            col.name = 'a' + str(i)
            col.kind = int  # Set the variable type to integer
            col.bounds = 0, 1   # Set the bounds to 0 and 1

            # Relevance score
            lp.obj[i] = 1/(i+1) + get_cosine_similarity(sentences[i], doc_corpus_vector)

        for i in range(num_sentences):
            for j in range(i+1, num_sentences):
                # Get the (i,j) variable index
                var_index = int(num_sentences + i * (num_sentences - 1) - (i * (i - 1)) / 2 + j - i - 1)

                # Redundancy score
                lp.obj[var_index] = get_cosine_similarity(sentences[i], sentences[j])
                col = lp.cols[var_index]
                col.name = 'a' + str(i) + str(j)
                col.kind = int

                # If the redundancy is 0, set the bounds to 0. This reduces number of operations for ILP to perform
                if lp.obj[var_index] > 0:
                    col.bounds = 0, 1
                else:
                    col.bounds = 0, 0

        num_rows = int(1 + 3*num_sentences*(num_sentences - 1)/2)
        lp.rows.add(num_rows)
        mat = np.zeros((num_rows, num_cols), dtype=int)
        curr_row = 0

        # Length bucket constraint
        lp.rows[curr_row].name = 'len'
        lp.rows[curr_row].bounds = 0, K
        mat[curr_row,:num_sentences] = [sentence_token_len[(article_id,i)] for i in range(num_sentences)]

        # Inclusion and inverse constraint
        for i in range(num_sentences):
            for j in range(i+1, num_sentences):
               
                curr_row += 1
                lp.rows[curr_row].name = 'mutual' + str(i)
                lp.rows[curr_row].bounds = -1, 0

                var_index = int(num_sentences + i * (num_sentences - 1) - (i * (i - 1)) / 2 + j - i - 1)

                # Only set the constraint for non-zero redundancy
                if lp.obj[var_index] > 0:
                    mat[curr_row,i] = -1
                    mat[curr_row,var_index] = 1

                curr_row += 1
                lp.rows[curr_row].name = 'mutual' + str(j)
                lp.rows[curr_row].bounds = -1, 0

                # Only set the constraint for non-zero redundancy
                if lp.obj[var_index] > 0:
                    mat[curr_row,j] = -1
                    mat[curr_row,var_index] = 1

                curr_row += 1
                lp.rows[curr_row].name = 'mutual' + str(i) + str(j)
                lp.rows[curr_row].bounds = 0, 1

                # Only set the constraint for non-zero redundancy
                if lp.obj[var_index] > 0:
                    mat[curr_row,i] = 1
                    mat[curr_row,j] = 1
                    mat[curr_row,var_index] = -1

        # Change numpy to a list of tuples
        rows, cols = np.where(mat != 0)
        constraint_mat = [(int(row),int(col),mat[row,col]) for row,col in zip(rows,cols)]

        lp.matrix = constraint_mat

        # Solve the ILP problem
        lp.simplex()
        lp.intopt()     # Perform integer optimization

        # Ensure that optimal solution is found
        if lp.status != 'opt':
            assert False, "Solution is not optimal!"

        # Get variables set to one
        relevant_sentences = [c.primal for c in lp.cols[:num_sentences]]

        # Write the summary to the file
        f.write(str(article_id) + " :")
        for i, sentence in enumerate(relevant_sentences):
            if sentence == 1:
                f.write(" " + " ".join(articleDict[article_id][i]))

        f.write("\n")

    f.close()
    print("ILP problem solved!")

if __name__ == "__main__":
    if len(sys.argv) != 2 :
        print("Error! Usage: python3 Assignment3_21CS10057_summarizer.py <path to data file>") 
        sys.exit(1)

    dataset_path = sys.argv[1]
    articleDict = parseDataset(dataset_path)

    inverted_index = get_inverted_index(articleDict)

    # Prepare a dictionary of document frequency (DF) for each term in the inverted index
    document_frequency = {term : len(postings) for term, postings in inverted_index.items()}

    # Prepare a (term,document) matrix for term frequency (TF) for each term-doc pair in the inverted index
    doc_term_frequency = {}
    for term, postings in inverted_index.items():
        for (article_id,article_cnt) in postings:
            doc_term_frequency[(term,article_id)] = article_cnt

    # Get total documents in the collection
    total_docs = set(doc_id for (_, doc_id) in doc_term_frequency.keys())
    print("Total articles in the collection:", len(total_docs))

    # Get total terms in the collection
    total_terms = [term for term in inverted_index.keys()]
    print("Total terms in the collection:", len(total_terms))

    # Prepare a dictionary of sentence lengths
    sentence_token_len = {}

    # Perpare a term frequency (TF) matrix for each term-sentence pair
    sentence_term_frequency = {}
    for article_id, sentences in articleDict.items():
        sentence_term_frequency[article_id] = {}

        for i, sentence in enumerate(sentences):
            sentence_term_frequency[article_id][i] = {}

            for token in sentence:
                if sentence_term_frequency[article_id][i].get(token) is None:
                    sentence_term_frequency[article_id][i][token] = 1
                else:
                    sentence_term_frequency[article_id][i][token] += 1
            sentence_token_len[(article_id,i)] = len(sentence)

    doc_corpus_vector = tf_idf_docs(doc_term_frequency, total_docs, total_terms)

    doc_sentence_vector_dict = tf_idf_sentences(sentence_term_frequency, document_frequency, total_docs, total_terms)

    relevant_sentences = ILPFunction(doc_corpus_vector, doc_sentence_vector_dict,sentence_token_len, articleDict)



    


