import csv
import numpy as np
import sys
from collections import Counter, defaultdict

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An N x 2 np.ndarray. N is the number of data points in the tsv file. The
        first column contains the label integer (0 or 1), and the second column
        contains the movie review string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_dictionary(file):
    """
    Creates a python dict from the model 1 dictionary.

    Parameters:
        file (str): File path to the dictionary for model 1.

    Returns:
        A dictionary indexed by strings, returning an integer index.
    """
    dict_map = np.loadtxt(file, comments=None, encoding='utf-8',
                          dtype=f'U{MAX_WORD_LEN},l')
    return {word: index for word, index in dict_map}


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file for model 2.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map

def map_words_to_row(dict_input, X, row, word):
    if word in dict_input:
        idx = dict_input[word]
        X[row, idx + 1] = 1

def BagOfWordsDense(dict_input, input_data, f):
    word_dim = len(dict_input)
    data_dim = len(input_data)
    X = np.zeros((data_dim, word_dim + 1), dtype=int)
    
    for i in range(data_dim):
        X[i, 0] = input_data[i][0]
        counter = Counter(input_data[i][1].split())
        for word in counter.keys():
            map_words_to_row(dict_input, X, i, word)
    np.savetxt(f, X, fmt="%d", delimiter='\t', newline='\n', encoding=None)
    print("Total Number of Features:", X.shape[1] - 1)


def BagOfWordsSparse(dict_input, input_data, filename):
    word_dim = len(dict_input)
    data_dim = len(input_data)
    f = open(filename, "w")
    
    for i in range(data_dim):
        f.write("{}".format(input_data[i][0])) # label
        counter = Counter(input_data[i][1].split())
        for word in counter.keys():
            if word in dict_input:
                idx = dict_input[word]
                f.write(" {}:{}".format(idx, 1)) # idx:value
        f.write("\n")
    f.close()


def map_embedding_to_row(dict_input, X, row, kv):
    if kv[0] in dict_input:
        # print(kv)
        X[row][1:] += dict_input[kv[0]] * kv[1]
        return kv[1]
    return 0

def WordEmbeddings(feature_dictionary_input, input_data, f):
    word_dim = 300
    data_dim = len(input_data)
    X = np.zeros((data_dim, word_dim + 1), dtype=float)
    for i in range(data_dim):
        X[i, 0] = input_data[i][0]
        counter = Counter(input_data[i][1].split())
        cnt = 0
        for kv in counter.items():
            cnt += map_embedding_to_row(dict_input, X, i, kv)
        X[i][1:] /= cnt
    np.savetxt(f, X, fmt="%.6f",delimiter='\t', newline='\n', encoding=None)


def output_formatted_data(dict_input, input_data, output_filename, matrix_type):
    if matrix_type == "sparse":
        BagOfWordsSparse(dict_input, input_data, output_filename)
    else:
        BagOfWordsDense(dict_input, input_data, output_filename)


if __name__ == '__main__':
    train_input= load_tsv_dataset(sys.argv[1]) # shape(N,)
    dict_filename = sys.argv[2]
    formatted_train_out = sys.argv[3]
    matrix_type = sys.argv[4]

    dict_input = load_dictionary(dict_filename)
    
    output_formatted_data(dict_input, train_input, formatted_train_out, matrix_type)

# python3 src/python_scripts/preprocessing.py data/small_train_data.tsv data/dict.txt data/formatted_small_train_sparse.tsv sparse