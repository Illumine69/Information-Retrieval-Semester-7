import sys
import nltk
import spacy
import string
import re
import os

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

def extract_id_and_text(query_file):
    '''
    Function: Extract the query ID and text in the .W field. 
              Process the text to get tokens.
    Output:   Returns a dictionary of id(keys) and tokens
    '''
    query_dict = {}
    regex = re.compile(r'(\.I)((?:(?!\n\.(?:I|T|A|W|B)).)+)(\n\.W\n)((?:(?!\n\.(?:I|T|A|W|B)).)+)', re.DOTALL)

    with open(query_file, 'r') as file:
        text = file.read()

        # match the text with regex
        text = regex.findall(text)
        i = 0
        while i < len(text):
            if text[i][0] == '.I':
                query_id = int(text[i][1].strip())
                print("Processing query_id: " + str(query_id) + "\n")
                
                query_text = text[i][3].strip()
                query_tokens = preprocess_text(query_text)
                query_dict[query_id] = query_tokens

            i += 1
        
    return query_dict


if __name__ == "__main__":
    # Error check to ensure correct python command
    if len(sys.argv) != 2:
        print("Error! Usage: python Assignment1_21CS10057_parser.py <path to the query file>")
        sys.exit(1)

    # Path to the query file
    query_file_path = sys.argv[1]

    # Store curent directory path
    DIRNAME = os.path.dirname(__file__)

    query_dict = extract_id_and_text(query_file_path)
    with open(f"{DIRNAME}/queries_21CS10057.txt", 'w') as file:
        for query_id, query_tokens in query_dict.items():
            query_text = ' '.join(query_tokens)
            file.write(f"{query_id}\t{query_text}\n")
    
    print("Saved queries as queries_21CS10057.txt file\n")
    
