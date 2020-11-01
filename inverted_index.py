from bs4 import BeautifulSoup
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import math
import numpy
import pickle

# This file creates an inverted index give an input file containing several documents

nltk.download('punkt')

# Extract text from a file and return as an array of documents
# A file consists of several documents
def extract(filename = ''):
    file = open(filename, 'r')
    soup = BeautifulSoup(file, "html.parser")
    docs=soup.findAll("doc")
    title_list=[x["title"] for x in docs]
    doc_list = [BeautifulSoup(str(doc), "html.parser").get_text() for doc in docs]
    return doc_list, title_list

# Write extracted text to a file
def write_to_file():
    with open('AB_10.txt','w') as f:
        for i in range(len(title_list)):
            f.write("\n******************\n%d.   %s \n %s\n" % (i,title_list[i], list_of_documents[i]))

# print(title_list)
# print(list_of_documents[0])

# Pre-process the extracted text
def pre_processing(text):
    tokens = []
    punc = [w for w in string.punctuation if not (w=='\'' or w=='-' or w=='%' or w==':')]
    punc = ''.join(punc)
    table = str.maketrans('', '', punc)
    for doc in text:
        # Tokenize text
        token_doc = word_tokenize(doc)
        # Remove punctuations
        token_doc = [w.translate(table) for w in token_doc]
        # Convert to lower case
        token_doc = [x.lower() for x in token_doc]
        token_doc = [w for w in token_doc if not (w=="''" or w=='' or w=="' '")]
        tokens.append(token_doc)
        # print(token_doc)
    return tokens

# print(tokens)

# Return unigrams from text
def get_unigrams(text):
    unigrams = [Counter(doc) for doc in text]
    return unigrams

# print(freq)

# create set from list of tokenized docs
def get_inverted_index(freq_list):
    
    inverted_index = {}
    for i in range(len(freq_list)): # Range over number of docs
        doc = freq_list[i]
        for key in doc.keys():
            if key in inverted_index.keys():
                inverted_index[key].append((i, doc[key]))
            else:
                inverted_index[key] = []
                inverted_index[key].append((i, doc[key]))

    return inverted_index

def generate_inverted_index(input_file):
    list_of_documents, title_list = extract(filename = input_file)
    processed_text = pre_processing(list_of_documents)
    freq = get_unigrams(processed_text) # Stores document-wise frequency of unigrams
    inverted_index = get_inverted_index(freq)

    with open('index_data/list_of_documents.data', 'wb') as f:
        pickle.dump(list_of_documents, f)

    with open('index_data/title_list.data', 'wb') as f:
        pickle.dump(title_list, f)

    with open('index_data/processed_text.data', 'wb') as f:
        pickle.dump(processed_text, f)

    with open('index_data/freq.data', 'wb') as f:
        pickle.dump(freq, f)

    with open('index_data/inverted_index.data', 'wb') as f:
        pickle.dump(inverted_index, f)
