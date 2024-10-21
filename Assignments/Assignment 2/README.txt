Roll No.: 21CS10057
Name: Sanskar Mittal

CS60092: Assignment 2 - Scoring and Evaluation

Requirements:
- Python 3.10.12
- NLTK library
- spaCy library with 'en_core_web_trf' model

Preprocessing pipeline:
1. Extract Doc IDs and tokens from .csv file
2. Replace hyphens with space for hyphenated words
3. Tokenization using nltk after lowercasing the text
4. Remove punctuations, numbers and stopwords
5. Lemmatization using spacy's 'en_core_web_trf' model
6. Stemming using nltk's PorterStemmer 

Assignment2_21CS10057_ranker.py :
    - The program takes the path to the data folder and the path to the bin file as input.
    - Ensure that the 'topics-rnd5.xml' file is present in the data folder.
    - It loads the bin file and the data folder.
    - The query file is then parsed to store query id and coresponding query.
    - The program uses the bin file to get the document frequency of each of the terms in the corpus.
    - It then uses the inverted index to get the term frequency of each term in each doc and query.
    - The program then uses the document frequency and term frequency to calculate the tf-idf score for each term in each doc and query, based on the different themes (namely lnc.ltc, lnc.Ltc and anc.apc), and creates the vectors wrt to the terms.
    - Dot product is then taken for each pair of doc id and query and for each query, the top 50 doc ids based on the cosine similarity are selected and saved with names as described in the assignment problem description.

Assignment2_21CS10057_evaluator.py :
    - The golf standard file is parsed to get relevance sscores for each query and doc id.
    - For average precision, a relevant document is considered to have a score of > 0 and non-relevant to either not be scored or a score of 0.
    - The program then calculates the average precision at 10 and 20, and the normalized discounted cumulative gain at 10 and 20 for each query.
    - Some queries have no relevant documents in top 10 or 20, so ideal dcg is zero. In these cases, the ndcg is considered to be zero.

Extra info:
- Vocabulary size:  8529
- Number of documents:  1154. Some documents had no abstracts, so they were discarded
- The inverted index is modified to store the term frequency of each term in each doc. Hence the postings list are of format: 
    > token: (doc_id1, term_freq1), (doc_id2, term_freq2), ...
- The metrics are rounded upto 5 decimal places
- log base 10 calculations have been for tf-idf scores
- Dataset Used: https://drive.google.com/file/d/1yE_eyCWI336ELjDO9Ysylgkxy0ip3pN8/view
- Assignment2_21CS10057_indexer.py has also been supplied along with model_queries_21CS10057.bin to get the inverted index. Please run the indexer file to update the bin file with the inverted index at any time.

Using 'en_core_web_trf' :
- This module is large in size and takes some time but has advantage to take more context into play for lemmatization
