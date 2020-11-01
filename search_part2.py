from search_part1 import *
import search_part1
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

# This file implements a retrieval system based on lnc-ltc scoring scheme...
# but along with the query terms, it also searches for the synonyms of the query term
# For example, Query: "U.S.A."
# System looks for: "U.S.A.", "U.S.", "United States of America", "USA", "US" etc.

def initialize_data_structures_1():
    
    global list_of_documents
    global title_list
    global processed_text
    global freq
    global inverted_index
    
    list_of_documents = search_part1.list_of_documents
    title_list = search_part1.title_list
    freq = search_part1.freq
    processed_text = search_part1.processed_text
    inverted_index = search_part1.inverted_index

# Computes the synonym set of all the terms in a query
def find_syn_set(query_tokens):
    
    lemmed_words=[]

    for word in query_tokens:
        synset = [word]
        for l in wordnet.synsets(word):
            for w in l.lemma_names():
                synset.append(w)
        lemmed_words.append(list(set(synset)))
    
    return lemmed_words

# Return an unsorted list of cosine similarity scores
def compute_scores_unsorted(query_wt, document_wt):
    
    scores = [[i, 0] for i in range(len(document_wt))]

    for i in range(len(document_wt)):
        
        doc_tf = document_wt[i]

        score = 0

        for j in range(len(doc_tf)):
            term = doc_tf[j][0]
            term_weight = get_query_term_weight(term, query_wt)

            score += term_weight*doc_tf[j][1]

        scores[i] = [i, score]

    # print(scores[:10])

    return scores    

# Merge existing scores and scores computed for a synonym set of a term
# w1: weight associated with score of original query term
# w2: weight associated with score of all the synonyms of original term
def merge_scores(scores, w1, w2):

    n = len(scores)
    org_score = scores[0]
    final_scores = [ [i, org_score[i][1]*w1] for i in range(len(org_score))]

    if n == 1:
        return final_scores
    else:
        for i in range(len(final_scores)):
            for j in range(1, n):
                final_scores[i][1] += (w2 * (scores[j])[i][1])

        return final_scores

# Function accepts user query, and ranks documents based on cosine similarity of query terms and their synonyms, then prints the top 10 results based on the rank
def modified_search():

    query = input('Enter your query: ')
    query_processed = query_pre_process(query)
    query_terms = get_query_terms(query_processed)
    print("Query Terms: ", query_terms)

    scores = []

    query_wt = get_normalized_query_scores(query_terms, freq, inverted_index)
    document_wt = get_normalized_doc_weights(query_terms, freq, inverted_index)

    # Compute cosine scores with sorting
    score = compute_scores_unsorted(query_wt, document_wt)
    scores.append(score)

    query_synonyms = find_syn_set(query_terms)
    for syn_set in query_synonyms:
        
        m_query = Counter(syn_set)
        q_wt = get_normalized_query_scores(m_query, freq, inverted_index)
        d_wt = get_normalized_doc_weights(m_query, freq, inverted_index)

        additional_score = compute_scores_unsorted(q_wt, d_wt)
        # Weight of the synonym scores can be modified.
        # Based on manual performance evaluation, the weight has been set to 0.2

        # Final_Score = Original_Term_Score + factor*Synonyms_Score
        scores.append(additional_score)

    final_score = merge_scores(scores, w1=1, w2=0.2)

    # Sort the final score
    score = sorted(final_score, key=lambda x: x[1], reverse=True)

    print("The top 10 documents matching with the query '", query, "' are:")
    
    for i in range(10):
        if i == len(title_list):
            break
        print(str(i) + ". Document: " + str(score[i][0]) + ". " + str(title_list[score[i][0]]) + ", Score: " + str(round(score[i][1], 3)))
