RESOURCES:

    1)   https://unix.stackexchange.com/questions/149965/how-to-copy-merge-two-directories

             rsync to sync train and test files in just 2 directories.

    2)   https://stackoverflow.com/questions/2407398/how-to-merge-lists-into-a-list-of-tuples

             make a list of tuples out of multiple lists

    3)   https://stackoverflow.com/questions/10695139/sort-a-list-of-tuples-by-2nd-item-integer-value

             sorting a list of tuples by an element of the tuple

    4)   https://stackoverflow.com/questions/3308102/how-to-extract-the-n-th-elements-from-a-list-of-tuples

             extracting elements of tuples from a list of tuples

LARGER INSTANCE:

    With all the movie reviews and my code I was over the 10 GB limit of my original instance when I
    started to add to my git folder.  So, I created a new instance with more memory and more storage
    capacity.

KNOWN BUGS:

    1)   None

ASSUMPTIONS:

    1)   None

ISSUES:

    1)   This will predict the correct name to unredact if the redacted name is in the train files
         and the similarity rank of the file to which the correct name belongs is high enough to
         not be cut off.  There are times when either:  a) the name does not exist in the train
         files, or b) the name belongs to a file that has a similarity rank beyond the cutoff.
         In either of the 2 exceptions the correct name will not be predicted.

MODULES:

    1)   makeTest.py
             a) import argparse
             b) import io
             c) import os
             d) import sys
             e) from pathlib import Path
             f) import shutil
             g) import re
             h) from nltk import sent_tokenize
             i) from nltk import word_tokenize
             j) from nltk import pos_tag
             k) from nltk import ne_chunk
             l) import glob
             m) from operator import itemgetter
             n) from nltk.tokenize import WhitespaceTokenizer
    2)   unredactor.py
             a) import argparse
             b) import io
             c) import os
             d) import pdb
             e) import sys
             f) import re
             g) from nltk import sent_tokenize
             h) from nltk import word_tokenize
             i) from nltk import pos_tag
             j) from nltk import ne_chunk
             k) from nltk import WhitespaceTokenizer
             l) import glob
             m) import pathlib
             n) import ast
             o) from sklearn.feature_extraction.text import TfidfVectorizer
             p) from sklearn.metrics.pairwise import cosine_similarity
             q) from operator import itemgetter

LARGE MOVIE REVIEW DATA SET:

    After unzipping the contents of the zipped file aclImdb_v1.tar.gz I kept only the pos and neg
    directories and their contents from the test and train directories.  I then combined the pos
    and neg directories creating only 2 directories, test and train.  These 2 directories contain
    all the movie reviews.

    I may refer to the files in the test directory as test file bank and the files in the train
    directory as the train file bank.

HOW TO USE:

    1)   makeTest.py

         This is an optional step.  This is used to create redacted files from the test file bank.

         I have done this using the following:

             pipenv run python project2/makeTest.py --num 20 --redact
             "LMRDS/aclImdb/test/1406_4.txt" --redact "LMRDS/aclImdb/test/*.txt"

         This passes the test file bank contents to the makeTest.py module.  For each of the
         files, the names are extracted and saved to a file in the directory:
         LMRDS/redacted/testRedNames.  In addition each of the names is redacted from the file
         and the file is saved in the directory:  LMRDS/redacted/testRedFiles.

         The flag num allows the user to limit how many of the test back files to extract names,
         redact, and save to a file.  The current implementation will take the first num files.

    2)   unredactor.py

         Since there are already redacted test files and name files saved in
         LMRDS/redacted/testRedFiles and testRedNames, respectively, we can skip to the
         unredactor by using the following:

             pipenv run python project2/unredactor.py --numCand 50 --numTrain 100 --redacted
             "LMRDS/redacted/testRedFiles/*.txt"

         This passes all the redacted files in the LMRDS/redacted/testRedFiles directory to the
         unredactor.py module.  The flag numCand limites the number of candidates to keep and
         the numTrain limits the number of training files to use.  The current implementation
         will take the first numTrain files.

HOW IT WORKS:

    1)   For each file in the test files passed and read, it finds the redacted names by
         searching for the redaction character, u2588
    2)   Once these are found features are extracted:
             a) total length of name
             b) number of tokens in name
             c) length of first name
             d) length of last name
    3)   For each of the first numTrain train files in the directory LMRDS/aclImdb/train names
         are found and redacted from the files
    4)   After removing stopwords from both the test file and the train files, TFIDF vectors are
         created for each of the files and then cosine similarity is found for each of the train
         files compared to the test file and ranked.
    5)   Starting with the most similar train file check if there are candidate names with the
         same features extracted from the test file's redacted names.  If there are, those names
         are added to the list of candidate names.
    6)   Using accuracy @k calculate the accuracy of this approach.  For this implementation,
         k = 1, 5, and 10
         a)   This first prints the accuracy for the file, then for all test files considered in
              aggregate.
         b)   Considered to be a true positive if the correct name is predicted within the top k
              ranked predictions.

WHY I CHOSE THIS APPROACH:

    The features that I thought would be important, and available in a redacted file, were the
    features I mentioned above:
         1) total length of name
         2) number of tokens in name
         3) length of first name
         4) length of last name

    And without these matching, the name will definitely not be predicted correctly, unless there
    is some sort of misspelling accounted for.

    But, there could be a lot of ties with these features.  So, to break the tie I used the
    cosine similarity on the TFIDF vector.

OUTPUT OF UNREDACTED FILES:

    The unredacted files are sent to the directory LMRDS/unredacted/testUnredFiles and saved with
    the same name.

HOW TO TEST:

    I tested the file 1406_4.txt

    Use the following to test:

        pipenv run python -m pytest

    The test_mine.py test file has 9 test methods:

        1) test_readFile_sanity()
            This tests to see if the readFile method has a length > 0
        2) test_do_extraction_sanity()
            This tests to see if do_extraction returns a dict
        3) test_do_extraction_length()
            This tests to see if the number of extracted names in 1406_4.txt is = 10
        4) test_redact_sanity()
            This tests to see if redact returns a dict
        5) test_extRed_value()
            This tests to see if the 1st redaction in 1406_4.txt is = to 4 of u2588
        6) test_extRed_length()
            This tests to see if there were 19 redactions in 1406_4.txt
        7) test_extFeatures_values()
            This tests to see if the extreacted feature of the first redaction of 1406_4.txt =
            [4, 1, 4, 4]
        8) test_extFeatures_lenth()
            This tests to see if there are 19 lists of feature extractions in 1406_4.txt
        9) test_similarityFinder_trivial()
            This tests to see if the file 1406_4.txt is similar to 1406_4.txt
