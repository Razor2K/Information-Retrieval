from search_part1 import search
from search_part2 import modified_search
from bm25 import search_BM25
from search_part1 import read_data_structures_1
from search_part2 import initialize_data_structures_1
from bm25 import initialize_data_structures_2

# Function to compare results of all the three approaches

def test_queries():

    # Reads all the data related to inverted index and necessary for computation
    # location: name of folder where data is stored
    read_data_structures_1(location = "index_data")
    # Initializes data structures for other models
    initialize_data_structures_1()
    initialize_data_structures_2()
    
    while(1):
        print("\n***************\nChoose one of the following options: \n")
        print("1. To search using lnc-ltc scoring scheme (Part-1)")
        print("2. Use a modified search that also considers synonyms of entered query (Part-2)")
        print("3. To search using the Okapi BM25 model (Part-2)")
        print("0. Exit")

        choice = int(input("Enter a valid integer: "))
        print("\n")
        if choice == 1:
            search()
        elif choice == 2:
            modified_search()
        elif choice == 3:
            search_BM25()
        elif choice == 0:
            break
        else:
            print("Input Valid. Re-enter")
