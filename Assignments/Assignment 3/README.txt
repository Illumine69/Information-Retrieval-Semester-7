Roll No.: 21CS10057
Name: Sanskar Mittal

CS60092: Assignment 3 - Summarization

Requirements:
- Python 3.10.12
- NLTK library
- spaCy library with 'en_core_web_sm' model
- glpk library along with pyglpk
- rouge-score module

Preprocessing pipeline:
1. Extract Article_IDs, text and summary from .csv file
2. Replace hyphens with space for hyphenated words
3. Tokenization using nltk after lowercasing the text
4. Remove punctuations, numbers and stopwords
5. Lemmatization using spacy's 'en_core_web_sm' model
6. Stemming using nltk's PorterStemmer 
7. Skip articles with highlight words > 200
8. Store the articles sentence-wise using sentence tokenizer

Assignment3_21CS10057_summarizer.py :
    - The program takes the path to the data file as input.
    - The data file is then parsed to store article id and corresponding sentences.
    - The program generates the document frequency of each of the terms in the corpus.
    - It then uses the inverted index to get the term frequency of each term in each article.
    - The program then uses the document frequency and term frequency to calculate the tf-idf score for each term in each doc, based on the themes (lnc.ltc), and creates the vectors wrt to the terms.
    - Document corpus vector is created by taking average of each document's vector.
    - A dictionary to store article-wise vectors for each sentence is created.
    - Finally ILP is used to select the sentences to form the summary.

Assignment3_21CS10057_evaluator.py :
    - The data file is parsed to get summary with article id
    - For each article, rouge-1 and rouge-2 scores are printed on the console

Extra info:
- Vocabulary size:  19682
- Number of articles: 999. Article number 500 had highlight words > 200 and hence was skipped.
- The inverted index is modified to store the term frequency of each term in each doc. Hence the postings list are of format: 
    > token: (doc_id1, term_freq1), (doc_id2, term_freq2), ...
- log base 10 calculations have been for tf-idf scores
- Dataset Used: https://drive.google.com/file/d/1UW-hA5xIeRbXYq541J1oyA6gThSFrnAA/view?usp=sharing

Using 'en_core_web_sm' :
- This module is small in size and used because file parsing and ILP calculation is comparatively faster.
