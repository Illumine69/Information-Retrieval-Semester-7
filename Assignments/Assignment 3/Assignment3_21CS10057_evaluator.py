import sys 
from nltk.stem import PorterStemmer
import spacy
import string
import pandas as pd
from rouge_score import rouge_scorer

# Maximum number of words in the summary
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
    Output: Highlights
    '''
    summaryDict = {}
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

        tokens = preprocess_text(summary)
        summaryDict[Docid] = ' '.join(tokens)

    print("Dataset parsed!")
    return summaryDict

def getSummary(summary_file):
    '''
    Returns a dictionary of generated summaries from the summary file
    '''

    summaryDict = {}
    with open(summary_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            Docid, summary = line.split(" : ")
            summaryDict[Docid] = summary.strip()

    return summaryDict

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 Assignment3_21CS10057_evaluator.py <path_to_data file> <path to Assignment3_21CS10057_summary.txt>")
        sys.exit(1)

    data_file = sys.argv[1]
    summary_file = sys.argv[2]

    goldSummaryDict = parseDataset(data_file)
    summaryDict = getSummary(summary_file)

    # Calculate ROUGE scores
    print("Calculating ROUGE scores...\n")
    for Docid in goldSummaryDict:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2']) 
        scores = scorer.score(goldSummaryDict[Docid], summaryDict[Docid])
        print("ROUGE scores for article %s: " % Docid)
        print("ROUGE-1: PRECISION: %f, RECALL: %f, F1: %f" % (scores['rouge1'].precision, scores['rouge1'].recall, scores['rouge1'].fmeasure))
        print("ROUGE-2: PRECISION: %f, RECALL: %f, F1: %f\n" % (scores['rouge2'].precision, scores['rouge2'].recall, scores['rouge2'].fmeasure))
