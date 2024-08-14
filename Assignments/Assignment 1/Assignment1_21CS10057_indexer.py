import sys
import nltk
import spacy
import string
import re
import os
import pickle

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

def extract_id_and_text(Folder_path):
    '''
    Function: Extract the document ID and text in the .W field. 
              Process the text to get tokens.
    Output:   Returns a dictionary of id(keys) and tokens
    '''
    file_name = f"{Folder_path}/CISI.ALL"
    doc_dict = {}
    regex = re.compile(r'(\.I|\n\.W\n)((?:(?!\n\.(?:I|T|A|W|X)).)+)', re.DOTALL)

    with open(file_name, 'r') as file:
        text = file.read()

        # match the text with regex
        text = regex.findall(text)
        i = 0
        while i < len(text)-1:
            if text[i][0] == '.I':
                j = i + 2
                doc_id = int(text[i][1].strip())
                print("Processing doc_id: " + str(doc_id) + "\n")
                doc_text = text[i+1][1].strip()
                doc_tokens = preprocess_text(doc_text)
                doc_dict[doc_id] = doc_tokens

                while j < len(text) and text[j][0] != '.I':
                    doc_text += " " + text[j-1][1].strip()
                    doc_tokens = preprocess_text(doc_text)
                    doc_dict[doc_id] = doc_tokens
                    j += 1
                i = j - 1
            i += 1
        
    return doc_dict

def get_inverted_index(doc_dict):
    '''
    Function: Creates an inverted index with tokens as keys and doc IDs as postings
    Output:   Returns a dictionary of token(keys) and id
    '''
    inv_index = {}
    for id, tokens in doc_dict.items():
        for token in tokens:
            if token not in inv_index:
                inv_index[token] = [id]
            else:
                if id not in inv_index[token]:
                    inv_index[token].append(id)

    # Sort the postings list
    for token in inv_index:
        inv_index[token].sort()

    return inv_index


if __name__ == "__main__":
    # Error check to ensure correct python command
    if len(sys.argv) != 2:
        print("Error! Usage: python Assignment1_21CS10057_indexer.py <path to the CISI folder>")
        sys.exit(1)

    # Path to the CISI folder
    cisi_folder_path = sys.argv[1]

    # Store cuurent directory path
    DIRNAME = os.path.dirname(__file__)

    document_dict = extract_id_and_text(cisi_folder_path)
    inv_dict = get_inverted_index(document_dict)
    
    pickle.dump(inv_dict, open(f"{DIRNAME}/model_queries_21CS10057.bin", "wb"))
    print("Saved inverted index as model_queries_21CS10057.bin file\n")
    
