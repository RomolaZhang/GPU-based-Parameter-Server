#include<iostream>
#include<vector>
#include<math.h>
#include<fstream>
#include<string>
#include <cuda.h>
#include <cuda_runtime.h>
#include<chrono>

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
#define SAFE_CALL(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define BLOCK_SIZE 256
#define MAX_DATA_SIZE 8
using namespace std;


struct Node {
    int idx;
    int label;
    int num_feature;
    int data_offset;
    int num_uncached;
    int uncached_offset;
};

double sigmoid(double x) {
    double e = exp(x);
    return e / (1 + e);
}

double forward(double* weights, vector<long long>& feature_ids, vector<double>& feature_vals,
               int start, int size) {
    double u = 0;
    for (int i = start; i < start + size; i++) {
        u += feature_vals[i] * weights[feature_ids[i]];
    }
    return sigmoid(u);
}

__device__ double device_sigmoid(double x) {
    double e = exp(x);
    return e / (1 + e);
}

__device__ void device_update_weights(double* parameter_cache, size_t num_parameter_cached, double* weights_from_cpu, long long* feature_ids, double* feature_vals,
                    double diff, double learning_rate, Node* meta_data, int sample_idx) {
    long long feature_id;

    size_t data_offset = meta_data[sample_idx].data_offset - meta_data[0].data_offset;
    size_t uncached_offset = meta_data[sample_idx].uncached_offset - meta_data[0].uncached_offset;
    size_t uncached = 0;
    // printf("i=%d[device_update_weights] batch_offset = %ld,num_feature=%d,[%d]\n", sample_idx, data_offset,meta_data[sample_idx].num_feature, feature_ids[data_offset]);

    for (int i = 0; i < meta_data[sample_idx].num_feature; i++) {
        feature_id = feature_ids[data_offset + i];
        if (feature_id < num_parameter_cached) {
            // // printf("device: feature_id = %ld < num_parameter_cached\n", feature_id);
            atomicAdd(&parameter_cache[feature_id], feature_vals[data_offset + i] * diff * learning_rate);
            // // printf("\nfeature_id=%d, parameter_cache[feature_id]=%f, val=%f\n", feature_id, parameter_cache[feature_id], feature_vals[data_offset + i]);
        } else {
            // // printf("device: feature_id = %ld > num_parameter_cached\n", uncached_offset + uncached);
            weights_from_cpu[uncached_offset + uncached] = feature_vals[data_offset + i] * diff * learning_rate;
            uncached++;
        }
    }
}

__device__ double device_forward(double* parameter_cache, size_t num_parameter_cached, double* weights_from_cpu,
            long long* feature_ids, double* feature_vals, Node* meta_data, int sample_idx) {
    double u = 0;
    long long feature_id;

    size_t data_offset = meta_data[sample_idx].data_offset - meta_data[0].data_offset;
    size_t uncached_offset = meta_data[sample_idx].uncached_offset - meta_data[0].uncached_offset;
    size_t uncached = 0;
    for (int i = 0; i < meta_data[sample_idx].num_feature; i++) {
        feature_id = feature_ids[data_offset + i];
        if (feature_id < num_parameter_cached) {
            u += feature_vals[data_offset + i] * parameter_cache[feature_id];
        } else{
            u += feature_vals[data_offset + i] * weights_from_cpu[uncached_offset + uncached];
            uncached++;
        }
    }
    return device_sigmoid(u);
}


__global__ void train_kernel(double* parameter_cache, size_t num_parameter_cached, double* weights_from_cpu, Node* meta_data, long long* feature_ids, double* feature_vals,
                double learning_rate, int batch_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0){
        for (int j = 0; j < 10; j++){
            // printf("feature_id = %ld\n", feature_vals[j]);
        }
    }
    if (i < batch_size) {
        double y_hat;
        double diff;
        //// printf("i = %d, feature_id_end = %ld\n", i, feature_ids[meta_data[i].data_offset + meta_data[i].num_feature - 1]);
        y_hat = device_forward(parameter_cache, num_parameter_cached, weights_from_cpu, feature_ids, feature_vals,
                               meta_data, i);
        diff = meta_data[i].label - y_hat;
        device_update_weights(parameter_cache, num_parameter_cached, weights_from_cpu, feature_ids, feature_vals,
                              diff, learning_rate, meta_data, i);
    }
}



double predict(double* weights, vector<Node>& meta_data, vector<long long>& feature_ids, vector<double>& feature_vals) {
    double y_hat;
    int pred_y;
    int size = meta_data.size();
    long long num_error = 0;
    for (int i = 0; i < size; i++) {
        y_hat = forward(weights, feature_ids, feature_vals,
                        meta_data[i].data_offset, meta_data[i].num_feature);
        pred_y = (y_hat >= 0.5)? 1 : 0;
        if (pred_y != meta_data[i].label) num_error++;
    }
    return (double) num_error / size;
}


void CPU_update_weights(size_t start_index, size_t batch_size, size_t num_uncached_weight,
                        double* weights, double* CPU_uncached_parameter_gradient,
                        vector<Node>& meta_data, vector<long long>& feature_ids) {
    // printf("CPU_update_weights: start_index=%d, batch_size=%d, num_uncached_weight=%d\n", start_index, batch_size, num_uncached_weight);
    int local_idx = 0, feature_ids_idx;
    for (int idx = start_index; idx < start_index + batch_size; idx++) {
        for (int f_idx = meta_data[idx].num_feature - meta_data[idx].num_uncached; f_idx < meta_data[idx].num_feature; f_idx++) {
            feature_ids_idx = meta_data[idx].data_offset + f_idx;
            int feature_id = feature_ids[feature_ids_idx];
            // printf("feature_id=%d, weight[i]= %f, local_idx=%d, g=%d\n", feature_id, weights[feature_id], local_idx, CPU_uncached_parameter_gradient[local_idx+1]);
            weights[feature_id] += CPU_uncached_parameter_gradient[local_idx++];
        }
    }
    // printf("\n\n");
}


size_t GPU_send(size_t start_idx, size_t num_parameter_cached,size_t array_space, double* weights, size_t& num_uncached_weight,
                vector<Node>& meta_data, vector<long long>& feature_ids, vector<double>& feature_vals, double* feature_weights_buffer,
                Node* next_meta_data, long long* next_feature_ids, double* next_feature_vals, double* next_feature_weights, 
                size_t& batch_size){
    num_uncached_weight = 0;
    size_t consumed_space = 0, current_idx = start_idx;
    size_t weight_idx = 0, feature_ids_idx;
    while (consumed_space < array_space && current_idx < meta_data.size()) {
        if (consumed_space + meta_data[current_idx].num_feature * MAX_DATA_SIZE > array_space)
            break;
        // num_cached_parameter_size = 200, num_samples = 4
        // num_features     [4, 3, 1, 2]
        // num_uncached     [3, 2, 0, 2]
        // f_idx        -    1    2    3     -   1      2     -    0      1
        // feature_ids_idx   1    2    3         5      6          8     9
        // weight_idx   -   0    1    2      -    3,  4,      -,   5,    6
        // feature_id  [- , 509, 999, 1000 | - , 405, 5000, | - | 3000, 5000]
        // feature_ids [33, 509, 999, 1000 | 29, 405, 5000, | 29| 3000, 5000]
        for (int f_idx = meta_data[current_idx].num_feature - meta_data[current_idx].num_uncached; f_idx < meta_data[current_idx].num_feature; f_idx++) {
            feature_ids_idx = meta_data[current_idx].data_offset + f_idx;
            int feature_id = feature_ids[feature_ids_idx];
            feature_weights_buffer[weight_idx] = weights[feature_id];
            weight_idx++;
        }
        consumed_space += meta_data[current_idx].num_feature * MAX_DATA_SIZE;
        num_uncached_weight += meta_data[current_idx].num_uncached;
        current_idx++;
    }

    if (start_idx < meta_data.size()) {
        size_t copy_size = (meta_data[current_idx].data_offset - meta_data[start_idx].data_offset) * MAX_DATA_SIZE;
        // printf("copy_size=%d\n", copy_size);
        cudaMemcpy(next_meta_data, &meta_data[start_idx], (current_idx - start_idx) * sizeof(Node), cudaMemcpyHostToDevice);
        cudaMemcpy(next_feature_ids, &feature_ids[meta_data[start_idx].data_offset], copy_size, cudaMemcpyHostToDevice);
        cudaMemcpy(next_feature_vals, &feature_vals[meta_data[start_idx].data_offset], copy_size, cudaMemcpyHostToDevice);

        // copy feature_weights
        cudaMemcpy(next_feature_weights, feature_weights_buffer, num_uncached_weight * MAX_DATA_SIZE, cudaMemcpyHostToDevice);

        batch_size = current_idx - start_idx;
    }
    // printf("GPU_send: start_idx=%d, current_idx=%d\n", start_idx, current_idx);
    return current_idx;
}


int main(int argc, char **argv) {
    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

    auto start = Clock::now();

    // Initialize all data structures
    string filename = argv[1];
    int total_features = stoi(argv[2]);
    long epoch = stol(argv[3]);
    double learning_rate = stod(argv[4]);


    double GPU_memory_size = stod(argv[5]); // in GB
    double GPU_cache_size = stod(argv[6]); // in GB

    // MAX_DATA_SIZE = 8
    // weights is of double type, and assume its size is at most 8 bytes (depending on platform)
    // so the max number of parameters is computed below
    size_t max_num_parameter_cached = GPU_cache_size * 1024 * 1024 * 1024 / MAX_DATA_SIZE;
    size_t num_parameter_cached = (total_features > max_num_parameter_cached)? max_num_parameter_cached: total_features;

    vector<Node> meta_data;
    vector<long long> feature_ids;
    vector<double> feature_vals;
    double* weights = (double*) malloc(sizeof(double) * total_features);

    cout << "Input filename: "<< filename << "\n";
    cout << "Number of features: "<< total_features << "\n";

    ifstream infile;
    infile.open(filename);

    string line, s_val;
    int label;
    long long f_id; 
    double f_val;
    int sample_idx = 0;
    int num_feature = 0;
    int data_offset = 0;
    int num_uncached = 0;
    int uncached_offset = 0;

    while (getline(infile, line)) {
        label = line[0] - '0';

        line = line.substr(2);
        std::size_t first, second;
        while ((first = line.find(":")) != string::npos) {
            f_id = stoll(line.substr(0, first));
            second = line.find(" ", first + 1);
            if (second == string::npos) {
                s_val = line.substr(first + 1);
            } else {
                s_val = line.substr(first + 1, second - first);
            }
            
            f_val = stod(s_val.c_str());
            feature_ids.push_back(f_id);
            feature_vals.push_back(f_val);
            num_feature++;
            // todo: check equal
            if (f_id > num_parameter_cached) num_uncached++;
            if (second == string::npos) break;
            line = line.substr(second);
        }

        Node node = {sample_idx, label, num_feature, data_offset, num_uncached, uncached_offset};
        data_offset += num_feature;
        uncached_offset += num_uncached;

        meta_data.push_back(node);
        sample_idx++;
        num_feature = 0;
        num_uncached = 0;
    }
    // Add end node
    Node node = {sample_idx, label, num_feature, data_offset, num_uncached, uncached_offset};
    meta_data.push_back(node);

    auto init_time = duration_cast<dsec>(Clock::now() - start).count();
    // printf("Initialization Time: %lf.\n", init_time);

    auto before_copy = Clock::now();
    // Compute number of blocks and threads per block
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((meta_data.size() + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Define kernel data structures
    Node *current_meta_data, *next_meta_data;
    long long *current_feature_ids, *next_feature_ids;
    double *current_feature_vals, *next_feature_vals;
    double *current_feature_weights, *next_feature_weights;
    double *GPU_parameter_cache, *CPU_uncached_parameter_gradient;

    // Compute kernel memory space
    size_t max_array_space = (GPU_memory_size * 1024 * 1024 * 1024 - num_parameter_cached * MAX_DATA_SIZE) / 8;
    
    size_t array_space = (feature_ids.size() * MAX_DATA_SIZE > max_array_space) ? max_array_space: feature_ids.size() * MAX_DATA_SIZE;
     
    size_t current_batch_size, next_batch_size;

    double *feature_weights_buffer = (double *) malloc(array_space);

    // Allocate device memory buffers on the GPU using cudaMalloc
    SAFE_CALL(cudaMalloc((void**)&current_meta_data, array_space));
    SAFE_CALL(cudaMalloc((void**)&current_feature_ids, array_space));
    SAFE_CALL(cudaMalloc((void**)&current_feature_vals, array_space));
    SAFE_CALL(cudaMalloc((void**)&current_feature_weights, array_space));

    SAFE_CALL(cudaMalloc((void**)&next_meta_data, array_space));
    SAFE_CALL(cudaMalloc((void**)&next_feature_ids, array_space));
    SAFE_CALL(cudaMalloc((void**)&next_feature_vals, array_space));
    SAFE_CALL(cudaMalloc((void**)&next_feature_weights, array_space));

    CPU_uncached_parameter_gradient = (double*) malloc(array_space);

    SAFE_CALL(cudaMalloc((void**)&GPU_parameter_cache, num_parameter_cached * sizeof(double)));

    auto before_train = Clock::now();

    size_t current_idx, next_idx, current_num_uncached_weight, next_num_uncached_weight;


    // run kernel
    for (int e = 0; e < epoch; e++) {
        current_idx = 0;
        next_idx = GPU_send(current_idx, num_parameter_cached, array_space, weights, current_num_uncached_weight,
                            meta_data, feature_ids, feature_vals, feature_weights_buffer,
                            current_meta_data, current_feature_ids, current_feature_vals, current_feature_weights, current_batch_size);
        while (current_idx < meta_data.size()){
            // printf("current_idx = %d\n", current_idx);
            int weight_start = current_idx, batch_size = next_idx - current_idx;
            current_idx = next_idx;
    
            // train model
            train_kernel<<<gridDim, blockDim>>>(GPU_parameter_cache, num_parameter_cached, 
                                                current_feature_weights, current_meta_data, current_feature_ids, current_feature_vals,
                                                learning_rate, current_batch_size);
            // Overlapping the execution of below function on the CPU with the kernel execution on the GPU
            next_idx = GPU_send(current_idx, num_parameter_cached, array_space, weights, next_num_uncached_weight,
                                meta_data, feature_ids, feature_vals, feature_weights_buffer,
                                next_meta_data, next_feature_ids, next_feature_vals, next_feature_weights, next_batch_size);
            

            // move weights back
            cudaMemcpy(CPU_uncached_parameter_gradient, current_feature_weights, current_num_uncached_weight * sizeof(double), cudaMemcpyDeviceToHost);
            
            CPU_update_weights(weight_start, batch_size, current_num_uncached_weight, weights, CPU_uncached_parameter_gradient,
                               meta_data, feature_ids);
                            
            current_meta_data = next_meta_data;
            current_feature_ids = next_feature_ids;
            current_feature_vals = next_feature_vals;
            current_feature_weights = next_feature_weights;
            current_batch_size = next_batch_size;
            current_num_uncached_weight = next_num_uncached_weight;
            
        }


    }


    auto train_time = duration_cast<dsec>(Clock::now() - before_train).count();
    printf("Training Time: %lf.\n", train_time);

    auto before_predict = Clock::now();


    // move weights back
    cudaMemcpy(weights, GPU_parameter_cache, num_parameter_cached * sizeof(double), cudaMemcpyDeviceToHost);


    double error = predict(weights, meta_data, feature_ids, feature_vals);
    auto predict_time = duration_cast<dsec>(Clock::now() - before_predict).count();
    printf("Prediction Time: %lf.\n", predict_time);


    printf("error(train): %f\n", error);

    cudaFree(current_meta_data);
    cudaFree(current_feature_ids);
    cudaFree(current_feature_vals);
    cudaFree(current_feature_weights);

    cudaFree(next_meta_data);
    cudaFree(next_feature_ids);
    cudaFree(next_feature_vals);
    cudaFree(next_feature_weights);

    cudaFree(GPU_parameter_cache);

    meta_data.clear();
    feature_ids.clear();
    feature_vals.clear();

    free(weights);
    return 0;
}
