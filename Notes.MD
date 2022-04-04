- Feature engineering of the small and medium dataset: specify dense or sparse format
```shell
Small / Large
Total Number of Features: 14164



python3 src/python_scripts/preprocessing.py data/small_train_data.tsv data/dict.txt data/formatted_small_train_sparse.tsv sparse


python3 src/python_scripts/preprocessing.py data/large_train_data.tsv data/dict.txt data/formatted_large_train_sparse.tsv sparse


python3 src/python_scripts/preprocessing.py data/large_train_data.tsv data/dict.txt data/formatted_large_train_dense.tsv dense

```

- Validate the correctness of logistic regression with python
```shell
# python3 src/python_scripts/lr.py data/formatted_large_train_dense.tsv dense 500 0.00001
# error(train): 0.042500

# python3 src/python_scripts/lr.py data/formatted_large_train_sparse.tsv sparse 500 0.00001
# error(train): 0.041667


# python3 src/python_scripts/lr.py data/formatted_small_train_sparse.tsv sparse 500 0.00001
# error(train): 0.011429

# python3 src/python_scripts/lr.py data/formatted_small_train_dense.tsv dense 500 0.00001
# error(train): 0.011429


# ./lr data/formatted_small_train_sparse.tsv 14164 500 0.00001
# Input filename: data/formatted_small_train_sparse.tsv
# Numeber of features: 14164
# error(train): 0.008571

# ./lr data/formatted_large_train_sparse.tsv 14164 500 0.00001
# Input filename: data/formatted_large_train_sparse.tsv
# Numeber of features: 14164
# error(train): 0.028333
```

- C++ based logistic regression
```shell
gcc lr.cc -o lr 

python3 src/python_scripts/lr.py data/formatted_small_train_dense.tsv dense 500 0.00001
error(train): 0.011429
```