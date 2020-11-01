from inverted_index import *
import nltk
import pickle

# Implementation of query retrieval system that uses lnc-ltc scoring scheme

# Reads all the data related to inverted index and necessary for computation
# location: name of folder where data is stored
def read_data_structures_1(location):
    
    global list_of_documents
    global title_list
    global processed_text
    global freq
    global inverted_index
    
    with open(location + '/list_of_documents.data', 'rb') as f:
        list_of_documents = pickle.load(f)

    with open(location + '/title_list.data', 'rb') as f:
        title_list = pickle.load(f)

    with open(location + '/processed_text.data', 'rb') as f:
        processed_text = pre_processing(list_of_documents)

    with open(location + '/freq.data', 'rb') as f:
        freq = get_unigrams(processed_text) # Stores document-wise frequency of unigrams

    with open(location + '/inverted_index.data', 'rb') as f:
        inverted_index = get_inverted_index(freq)

# Pre-process query
# Similar steps as the documents
def query_pre_process(query):
    punc = [w for w in string.punctuation if not (w=='\'' or w=='-' or w=='%' or w==':')]
    punc = ''.join(punc)
    table = str.maketrans('', '', punc)
    new_query = word_tokenize(query)
    new_query = [w.translate(table) for w in new_query]
    new_query = [x.lower() for x in new_query]
    new_query = [w for w in new_query if not (w=="''" or w=='' or w=="' '")]

    return new_query

# Return all the query terms with the corresponding count
def get_query_terms(query):
    return Counter(query)

# Generated normalized query scores
# Scoring scheme: ltc
# l -> Logarithmic tf
# t -> idf
# c -> Cosine normalization
def get_normalized_query_scores(query_terms, freq_list, inverted_index):

    # Calculate logarithmic tf
    tf_weights = {}

    for term in query_terms:
        tf_weights[term] = 1 + math.log10(query_terms[term])

    # Calculate idf
    idf = {}
    N = len(freq_list)
    for term in query_terms:
        if term in inverted_index.keys():
            idf[term] = math.log10( N / len(inverted_index[term]))
        else:
            idf[term] = 0

    query_tf_idf = {}

    for term in query_terms:
        query_tf_idf[term] = idf[term]*tf_weights[term]

    # Applying cosine normalization
    cos_factor = math.sqrt(sum([x**2 for x in query_tf_idf.values()]))
    
    if cos_factor != 0:
        cos_factor= 1/cos_factor

    for term in query_tf_idf:
        query_tf_idf[term] = cos_factor * query_tf_idf[term];

    return query_tf_idf

# Generate normalize document scores
# Scoring scheme: lnc
# l -> Logarithmic tf
# n -> No idf
# c -> Cosine normalization
def get_normalized_doc_weights(query_terms, freq_list, inverted_index):
    
    doc_weights = [[] for i in range(len(freq_list))]

    # Finding logarithmic tf
    for i in range(len(freq_list)):
        for term in freq_list[i].keys():
            val = freq_list[i][term]
            doc_weights[i].append([term, 1 + math.log10(val)])

    # Applying cosine normalization
    normalized_doc_weights = [[] for i in range(len(doc_weights))]

    for i in range(len(doc_weights)):
        doc_tf = doc_weights[i]

        square_sum = math.sqrt(sum( [v[1]**2 for v in doc_tf]))
        if square_sum != 0:
            factor = 1 / square_sum

        for j in range(len(doc_tf)):
            normalized_doc_weights[i].append([doc_tf[j][0], doc_tf[j][1]*factor])
    
    return normalized_doc_weights

# Function that returns the weight of a given term in query
# Returns 0 if term not in query
def get_query_term_weight(term, term_weights):
    if term in term_weights.keys():
        return term_weights[term]
    else:
        return 0

# Computes the cosine similarity between query and document weights
# Returns a sorted list of documents with their scores in non-increasing order
def compute_scores(query_wt, document_wt):

    scores = [[i, 0] for i in range(len(document_wt))]

    for i in range(len(document_wt)):
        
        doc_tf = document_wt[i]

        score = 0

        for j in range(len(doc_tf)):
            term = doc_tf[j][0]
            term_weight = get_query_term_weight(term, query_wt)

            score += term_weight*doc_tf[j][1]

        scores[i] = [i, score]

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # print(scores[:10])

    return scores

# Function accepts user query, and ranks documents based on cosine similarity, then prints the top 10 results based on the rank
def search():
    query = input('Enter your query: ')
    # Pre-process the query
    query_processed = query_pre_process(query)
    query_terms = get_query_terms(query_processed)
    print("Query Terms: ", query_terms)

    # Find query and docuemnt weights
    query_wt = get_normalized_query_scores(query_terms, freq, inverted_index)
    document_wt = get_normalized_doc_weights(query_terms, freq, inverted_index)

    # Find the ranking
    scores = compute_scores(query_wt, document_wt)

    print("The top 10 documents matching with the query '", query, "' are:")
    
    for i in range(10):
        if i == len(title_list):
            break
        print(str(i) + ". Document: " + str(scores[i][0]) + ". " + str(title_list[scores[i][0]]) + ", Score: " + str(round(scores[i][1], 3)))

