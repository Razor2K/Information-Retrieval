from search_part1 import *
import search_part1

# Rank documents using a probabilistic approach based on the Okapi BM25 model

def initialize_data_structures_2():
    
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

# Compute scores using the Okapi BM25 method
# Improving idf term by factoring in term frequency and document length
# k: tuning parameter controlling the document term frequency scaling
# b: tuning parameter controlling the scaling by document length
# Computes RSVd
def get_BM25(query_terms_list, k, b):
    N = len(list_of_documents)
    
    length_av = 0
    
    for doc in processed_text:
        length_av += len(doc)
    
    length_av /= N
    
    RSV_all_docs = [[i, 0] for i in range(0, N)]
    
    for i in range(0, len(freq)):
        # doc_freq = {}
        doc_freq = dict(freq[i])
        score = 0
        
        for term in query_terms_list:
            if term in doc_freq:
                df = len(inverted_index[term])
                tf = doc_freq[term]
                temp_score = math.log10(N/df)* (k+1)*tf
                length_doc = len(processed_text[i])
                temp_score /= k*((1-b) +b*length_doc/length_av) + tf 
                score+= temp_score
        
        RSV_all_docs[i]=[i, score]

    RSV_all_docs = sorted(RSV_all_docs, key=lambda x: x[1], reverse=True)
    
    return RSV_all_docs


# Function accepts user query, and ranks documents based on Okapi BM25 score, then prints the top 10 results based on the rank
def search_BM25():
    query = input('Enter your query: ')
    query_processed = query_pre_process(query)
    query_terms = get_query_terms(query_processed)
    print("Query Terms: ", query_terms)

    # Values for parameters k and b set to 0.5
    k=0.5
    b=0.5
    scores = get_BM25(query_terms, k, b)

    print("The top 10 documents matching with the query '", query, "' are:")
    
    for i in range(10):
        if i == len(title_list):
            break
        print(str(i) + ". Document: " + str(scores[i][0]) + ". " + str(title_list[scores[i][0]]) + ", Score: " + str(round(scores[i][1], 3)))
        