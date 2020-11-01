## Information Retrieval Assignment-1

Name: Komal Vasudeva
ID: 2017A7PS0103P

Name: Rohit Rajhans
ID: 2017A7PS0105P

Name: Mayank Jain
ID: 2017A7PS0179P

Instructions to run:

1. Make sure pip is installed to install the dependencies
2. Run `pip install -r requirements.txt` to install the dependencies
3. Run the file `code/main.py` to view the output

- 3 methods can be used to retrieve query results:
1. The first one uses the lnc-ltc scoring scheme without significant pre-processing as part of Part-1 of the assignment.

The part 2 required a more robust retrieval model. The two approaches used have been use of a dictionary and changing the scoring model.

2. The first modified approach computes the synonym sets of each of the query term and calculates the score for all the synonyms. A modifiable weight is assigned to the score generated from the synonym sets.

3. The second modified approach uses the Okapi BM25 model which is more robust. The value of parameters k and b in the formula are modifiable and are currently set to 0.5

- Code is divided in two parts:
1. `inverted_index.py`: Creates and saves the inverted index and other necessary information in index_data folder
2. `test_queries.py`, `search_part1.py`, `search_part2.py`, `bm25.py`: Implement the necessary retrieval models

`main.py` is the driver file that combine both the parts.