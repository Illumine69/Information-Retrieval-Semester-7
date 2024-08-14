Roll No.: 21CS10057
Name: Sanskar Mittal

CS60092: Assignment 1 - Inverted Index, Boolean Document Retrieval

Requirements:
- Python 3.10.12
- NLTK library
- spaCy library with 'en_core_web_trf' model
- regex

Preprocessing pipeline:
1. Extract Doc IDs and tokens from '.I' and '.W' fields using regular expressions
2. Replace hyphens with space for hyphenated words
3. Tokenization using nltk after lowercasing the text
4. Remove punctuations, numbers and stopwords
5. Lemmatization using spacy's 'en_core_web_trf' model
6. Stemming using nltk's PorterStemmer 

index.py:
- Text(after following the preprocessing pipeline) is added to a dictionary
- Using dictionary, an inverted index is built and stored as .bin file

parser.py:
- Text(after following the preprocessing pipeline) is added to a dictionary
- ID and corresponding query are stored in a text file

bool.py:
- For each query, take AND of each query token
- AND is done using a modified merge routine to find list of docID(s) having the query with help of postings list in inverted index
- NULL lists handling is also incorporated in the logic

results.txt:
- Contains space-separated list of doc IDs answering the query using the boolean retrieval
- Empty list indicates no matching doc ID.

Regex logic:
1. For indexer.py:  
    (\.I|\n\.W\n)((?:(?!\n\.(?:I|T|A|W|X)).)+):
    - (\.I|\n\.W\n) : This part of the regex defines a capturing group ( ... ) that matches either  .I  or  \n.W\n . The `|` character functions as an OR operator, meaning it will match either `.I` or `\n.W\n` when encountered.
                      \n\.W\n is used to ensure no wrong matching in cases like author's name having '.W' in the corpus. Newline starting and ending ensure only a single line with '.W'

    - ((?:(?!\n\.(?:I|T|A|W|X)).)+) : This part is another capturing group that matches the content following either `.I` or `\n.W\n` until it reaches the next field (.I, .T, .A, .W, .X) in the document.
        
        >  (?:(?!\n\.(?:I|T|A|W|X)).)+ : This inner expression is a non-capturing group (?: ... ) that matches any character (except a newline) one or more times. It does this by using "." to match any character and "+" to indicate one or more occurrences.
        >  (?!\n\.(?:I|T|A|W|X)) : This is a negative lookahead assertion (?! ... ), which checks that the following characters are not the start of a field (.I, .T, .A, .W, .X). It uses (?:I|T|A|W|X) to define a non-capturing group that matches any of these field names. 
                                   So, this part ensures that the content captured does not include another field name.
        
    Hence, this regular expression captures data in the format of .I (document ID) and .W (document text) fields while ensuring it doesn't accidentally include content from other fields (.A, .T, .X).

2. For parser.py:
    (\.I)((?:(?!\n\.(?:I|T|A|W|B)).)+)(\n\.W\n)((?:(?!\n\.(?:I|T|A|W|B)).)+):
    - This regex is almost same as the regex used in indexer.py
    - The only difference is that '.W' field is expected to follow immediately in next line after '.I' field
    - This is because CISI.QRY has some redundant text which are extracted as queries otherwise

Using 'en_core_web_trf' :
- This module is large in size and takes some time but has advantage to take more context into play for lemmatization
