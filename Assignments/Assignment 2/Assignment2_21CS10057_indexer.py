import sys
import nltk
import spacy
import string
import os
import pickle
import pandas as pd
import numpy as np

# Create a stemmer object
stemmer = nltk.stem.PorterStemmer()

# Load the space model
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

def extract_id_and_text(File_path):
    '''
    Function: Extract the document ID and text in the csv file 
              Process the text to get tokens.
    Output:   Returns a dictionary of id(keys) and tokens
    '''
    file_name = File_path
    doc_dict = {}

    df = pd.read_csv(file_name)
    # iterate over the rows to save two columns
    print("Extracting document ID and text...")
    for index, row in df.iterrows():
        doc_id = row[df.columns[0]]
        doc_text = row[df.columns[1]]

        print(doc_id)
        if doc_text is np.nan:
            continue

        doc_tokens = preprocess_text(doc_text)
        doc_dict[doc_id] = doc_tokens
    return doc_dict

def get_inverted_index(doc_dict):
    '''
    Function: Creates an inverted index with tokens as keys and doc IDs as postings
    Output:   Returns a dictionary of token(keys) and id
    '''
    inv_index_dict = {}
    for id, tokens in doc_dict.items():
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

    return inv_index

if __name__ == "__main__":
    # Error check to ensure correct python command
    if len(sys.argv) != 2:
        print("Error! Usage: python Assignment1_21CS10057_indexer.py <path to document file csv>")
        sys.exit(1)

    # Path to the document file
    document_path = sys.argv[1]

    # Store current directory path
    DIRNAME = os.path.dirname(__file__)

    document_dict = extract_id_and_text(document_path)
    inv_dict = get_inverted_index(document_dict)
    
    pickle.dump(inv_dict, open(f"{DIRNAME}/model_queries_21CS10057.bin", "wb"))
    print("Saved inverted index as model_queries_21CS10057.bin file\n")
    
