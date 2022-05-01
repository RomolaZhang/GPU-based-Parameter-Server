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
g++ src/lr.cc -std=c++11 -o lr 


python3 src/python_scripts/lr.py data/formatted_small_train_dense.tsv dense 500 0.00001
error(train): 0.011429
```


- CUDA based logistic regression
```shell

nvcc -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -std=c++11 -o cuda_lr src/cuda_lr.cu

./cuda_lr data/formatted_small_train_sparse.tsv 14164 500 0.00001
```

- Download Dataset
```bash
wget https://s3.amazonaws.com/cmucc-datasets/kdd10.gz
gunzip kdd10.gz 
```

- Previous Records
```bash
-bash-4.2$ ./cuda_lr data/formatted_large_train_sparse.tsv 14164 500 0.00001 2 1
Input filename: data/formatted_large_train_sparse.tsv
Number of features: 14164
Initialization Time: 0.112439.
CUDA Data Preparation Time: 0.069946.
Training Time: 0.000530.
Prediction Time: 0.232393.
error(train): 0.041667

-bash-4.2$ ./cuda_lr /usr/tmp/618-data/kdd10 20216830 1 0.00001
Input filename: /usr/tmp/618-data/kdd10
Number of features: 20216830
Initialization Time: 80.090016.
CUDA Data Preparation Time: 0.549704.
Training Time: 0.000022.
Prediction Time: 0.721991.
error(train): 0.143733

-bash-4.2$ ./cuda_lr /usr/tmp/618-data/kdd10 20216830 1 0.00001 2 1
Input filename: /usr/tmp/618-data/kdd10
Number of features: 20216830
Training Time: 0.753248.
Prediction Time: 0.518369.
error(train): 0.144061





-bash-4.2$ ./lr data/formatted_large_train_sparse.tsv 14164 500 0.00001
Input filename: data/formatted_large_train_sparse.tsv
Numeber of features: 14164
Initialization Time: 0.134746.
Training Time: 1.822311.
Prediction Time: 0.001623.
error(train): 0.041667



-bash-4.2$ ./lr /usr/tmp/618-data/kdd10 20216830 1 0.00001 
Input filename: /usr/tmp/618-data/kdd10
Numeber of features: 20216830
Initialization Time: 97.564933.
Training Time: 3.162660.
Prediction Time: 1.649220.
error(train): 0.143733

-bash-4.2$ ./cuda_lr /usr/tmp/kdd10 20216830 1 0.00001 4 1 1
Input filename: /usr/tmp/kdd10
Number of features: 20216830
Training Time: 0.027226.
Prediction Time: 0.399279.
error(train): 0.146991

-bash-4.2$ ./cuda_lr /usr/tmp/kdd10 20216830 1 0.00001 4 0 1
Input filename: /usr/tmp/kdd10
Number of features: 20216830
Training Time: 0.799217.
Prediction Time: 0.483909.
error(train): 0.146991

```


nvcc -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -std=c++11 -o cuda_lr src/cuda_lr.cu


./cuda_lr data/formatted_large_train_sparse.tsv 14164 500 0.00001 4 1 1

gdb ./cuda_lr
break main
run data/formatted_small_train_sparse.tsv 14164 1 0.00001 2 1



break cuda_lr.cu:302



- MPI

nvcc -O3 -m64 --gpu-architecture compute_61 -ccbin /usr/bin/gcc -std=c++11 -I/usr/local/depot/openmpi-4.0.2/include -L/usr/local/depot/openmpi-4.0.2/lib -lmpi -o mpi_cuda_lr src/mpi_cuda_lr.cu

mpirun -np 3 ./mpi_cuda_lr data/formatted_large_train_sparse.tsv 14164 500 0.00001 2 0.5 0.5

export LD_LIBRARY_PATH=/usr/local/depot/openmpi-4.0.2/lib