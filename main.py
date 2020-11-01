from inverted_index import *
from test_queries import test_queries

# Main file that generates and stores the inverted index and also calls the test_queries.py file to run the retrieval models

def main():

    print("Generating inverted index: ")
    generate_inverted_index(input_file="wiki_10")
    print("Inverted index stored")

    print("Test Queries: ")
    test_queries()

if __name__ == "__main__":
    main()