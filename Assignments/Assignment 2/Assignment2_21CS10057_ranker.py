import nltk
# from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import sys
import spacy
import pickle
import os
import math
import numpy as np

# Create a stemmer object
stemmer = PorterStemmer()

# Load the spacy model
nlp = spacy.load("en_core_web_trf")

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

def get_query_dict(data_folder_path):
    '''
    Function: Parse the query file and store query_id, query_text in a dictionary
    '''
    query_dict = {}
    query_file_path = os.path.join(data_folder_path, 'topics-rnd5.xml')

    topic_start_tag = "<topic number="
    topic_end_tag = "</topic>"
    query_start_tag = "<query>"
    query_end_tag = "</query>"

    xml_file_string = ""
    with open(query_file_path, 'r') as file:
        xml_file_string = file.read()

    print("Parsing the query file...")
    # Extract the query_id and query_text from the XML file
    while topic_start_tag in xml_file_string:
        # Find the start and end index of the topic
        topic_start_index = xml_file_string.index(topic_start_tag)
        topic_end_index = xml_file_string.index(topic_end_tag, topic_start_index)

        # Find the start and end index of the query
        query_start_index = xml_file_string.index(query_start_tag, topic_start_index)
        query_end_index = xml_file_string.index(query_end_tag, query_start_index)

        # Extract the query_id and query_text
        query_id = xml_file_string[topic_start_index + len(topic_start_tag):topic_end_index].split('"')[1]
        query_text = xml_file_string[query_start_index + len(query_start_tag):query_end_index].strip()

        # Preprocess the query text and store in the dictionary
        query_tokens = preprocess_text(query_text)
        query_dict[query_id] = query_tokens

        # Update the XML file string
        xml_file_string = xml_file_string[topic_end_index + len(topic_end_tag):]

    return query_dict

def tf_idf_docs(term_frequency, total_docs,total_terms, scheme=None):
    '''
    Function: Calculate the TF-IDF score for each term-document pair using ddd scheme
    Input: scheme is a list of 3 characters denoting the ddd scheme
    '''
    doc_tf_idf_vector = {}
    for doc in total_docs:
        doc_tf_idf_vector[doc] = {}

    max_tf = None
    if(scheme[0] == 'a'):
        max_tf = max([tf for tf in term_frequency.values()])

    # Iterate through the term frequency dictionary
    for (term, doc_id), tf in term_frequency.items():

        # Calculate the term frequency (TF) for the term-doc pair
        if(scheme[0] == 'l'):
            tf = 1 + math.log(tf, 10) if tf > 0 else 0
        elif(scheme[0] == 'a'):
            tf = 0.5 + 0.5 * tf / max_tf
        else:
            assert False, "Invalid scheme"

        tf_idf = None
        # Calculate the TF-IDF score for the term-doc pair
        if(scheme[1] == 'n'):
            tf_idf = tf * 1
        else:
            assert False, "Invalid scheme"

        # Store the TF-IDF score in the dictionary
        doc_tf_idf_vector[doc_id][term] = tf_idf

    # Add all terms so that each document vector is same size
    for doc_id in total_docs:
        for term in total_terms:
            if term not in doc_tf_idf_vector[doc_id].keys():
                doc_tf_idf_vector[doc_id][term] = 0

    # Normalize the TF-IDF scores
    if(scheme[2] == 'c'):
        for doc_id in total_docs:
            doc_vector = doc_tf_idf_vector[doc_id]
            doc_norm = math.sqrt(sum([score ** 2 for score in doc_vector.values()]))
            doc_tf_idf_vector[doc_id] = {term: score / doc_norm for term, score in doc_vector.items()}
    else:
        assert False, "Invalid scheme"

    return doc_tf_idf_vector

def tf_idf_queries(query_term_frequency,document_frequency, total_terms,total_docs,scheme=None):
    '''
    Function: Calculate the TF-IDF score for each term-query pair using the qqq scheme
    Input: scheme is a list of 3 characters denoting the qqq scheme
    '''
    query_tf_idf_vector = {}
    for query_id in query_term_frequency.keys():
        query_tf_idf_vector[query_id] = {}

    # Iterate through the query term frequency dictionary
    for query_id, query_terms in query_term_frequency.items():

        max_tf = None
        if(scheme[0] == 'a'):
            max_tf = max([tf for tf in query_terms.values()])

        avg_tf = None
        if(scheme[0] == 'L'):
            avg_tf = sum([tf for tf in query_terms.values()])/len(query_terms)
        
            
        for term, tf in query_terms.items():
            # Calculate the term frequency (TF) for the term-query pair
            if(scheme[0] == 'l'):
                tf = 1 + math.log(tf, 10) if tf > 0 else 0
            elif(scheme[0] == 'a'):
                tf = 0.5 + 0.5 * tf / max_tf
            elif(scheme[0] == 'L'):
                tf = (1 + math.log(tf, 10)) / (1 + math.log(avg_tf, 10)) if tf > 0 else 0
            else:
                assert False, "Invalid scheme"

            # Calculate the inverse document frequency (IDF) for the term
            idf = None
            if(scheme[1] == 't'):
                idf = math.log(len(total_docs) / document_frequency[term], 10)
            elif(scheme[1] == 'p'):
                idf = max(0, math.log((len(total_docs) - document_frequency[term]) / document_frequency[term], 10))
            else:
                assert False, "Invalid scheme"

            # Calculate the TF-IDF score for the term-query pair
            tf_idf = tf * idf

            # Store the TF-IDF score in the dictionary
            query_tf_idf_vector[query_id][term] = tf_idf

        # Add all terms so that each query vector is same size
        for term in total_terms:
            if term not in query_tf_idf_vector[query_id].keys():
                query_tf_idf_vector[query_id][term] = 0

    # Normalize the TF-IDF scores
    if(scheme[2] == 'c'):
        for query_id in query_term_frequency.keys():
            query_vector = query_tf_idf_vector[query_id]
            query_norm = math.sqrt(sum([score ** 2 for score in query_vector.values()]))
            query_tf_idf_vector[query_id] = {term: score / query_norm for term, score in query_vector.items()}
    else:
        assert False, "Invalid scheme"

    return query_tf_idf_vector

def get_cosine_similarity(query_tf_idf_vector, doc_tf_idf_vector):
    '''
    Function: Calculate the cosine similarity between each query and document
    '''
    cosine_similarity = {}
    for query_id in query_tf_idf_vector.keys():
        cosine_similarity[query_id] = {}

    # Iterate through the query TF-IDF vector
    for query_id, query_vector in query_tf_idf_vector.items():
        # Iterate through the document TF-IDF vector
        for doc_id, doc_vector in doc_tf_idf_vector.items():
            # Calculate the dot product between the query and document vectors
            dot_product = sum([query_vector[term] * doc_vector[term] for term in query_vector.keys()])

            # Calculate the cosine similarity between the query and document
            cosine_similarity[query_id][doc_id] = dot_product

    return cosine_similarity



if __name__ == "__main__":

    # Error check to ensure correct python command
    if len(sys.argv) != 3:
        print("Error! Usage: python Assignment2_21CS10057_ranker.py <path to data folder> <path_to_model_queries_21CS10057.bin>")
        sys.exit(1)

    data_folder_path = sys.argv[1] 
    bin_path = sys.argv[2]

    # Load the inverted index from the binary file 'model_queries.bin'
    print("Loading the inverted index...")
    with open(bin_path, 'rb') as index_file:
        inverted_index = pickle.load(index_file)

    # Parse the query file and story query_id, query_text in a dictionary
    query_dict = get_query_dict(data_folder_path)

    # Prepare a dictionary of document frequency (DF) for each term in the inverted index
    document_frequency = {term : len(postings) for term, postings in inverted_index.items()}

    # Prepare a (term,document) matrix for term frequency (TF) for each term-doc pair in the inverted index
    doc_term_frequency = {}
    for term, postings in inverted_index.items():
        for (doc_id,doc_cnt) in postings:
            doc_term_frequency[(term,doc_id)] = doc_cnt

    # Get total documents in the collection
    total_docs = set(doc_id for (_, doc_id) in doc_term_frequency.keys())
    print("Total documents in the collection:", len(total_docs))

    # Get total terms in the collection
    total_terms = [term for term in inverted_index.keys()]
    print("Total terms in the collection:", len(total_terms))

    # Perpare a term frequency (TF) matrix for each term-query pair in the query dictionary
    query_term_frequency = {}
    for query_id in query_dict.keys():
        query_tokens = query_dict[query_id]
        query_term_frequency[query_id] = {}
        for token in query_tokens:
            if token not in query_term_frequency[query_id]:
                query_term_frequency[query_id][token] = 1
            else:
                query_term_frequency[query_id][token] += 1

    schemes = [['l','n','c','l','t','c'], ['l','n','c','L','t','c'], ['a','n','c','a','p','c']]

    for i, scheme in enumerate(schemes):

        print(f"Calculating TF-IDF scores for list {chr(ord('A')+i)}...")
        # TF_IDF for documents
        print(f"Calculating TF-IDF scores for documents using scheme {scheme[:3]}...")
        doc_tf_idf_vector = tf_idf_docs(doc_term_frequency,total_docs,total_terms, scheme[:3])

        # TF_IDF for queries
        print(f"Calculating TF-IDF scores for queries using scheme {scheme[3:]}...")
        query_tf_idf_vector = tf_idf_queries(query_term_frequency,document_frequency, total_terms,total_docs, scheme[3:])

        # Calculate the cosine similarity between each query and document
        cosine_similarity = get_cosine_similarity(query_tf_idf_vector, doc_tf_idf_vector)

        # Store the query and top 50 docs in a file
        print(f"Storing the ranked list for list {chr(ord('A')+i)}...")
        with open(f"Assignment2_21CS10057_ranked_list_{chr(ord('A')+i)}.txt",'w') as f:
            for query_id, doc_scores in cosine_similarity.items():
                sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                top_docs = sorted_docs[:min(50, len(sorted_docs))]
                top_docs_string = ' '.join([str(doc_id) for doc_id,_ in top_docs])
                f.write(f"{query_id} : {top_docs_string}\n")
