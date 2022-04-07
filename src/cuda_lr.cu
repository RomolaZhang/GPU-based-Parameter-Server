#include<iostream>
#include<vector>
#include<math.h>
#include<fstream>
#include<string>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
using namespace std;

struct Node {
    int idx;
    int label;
    int num_feature;
    int data_offset;
};

vector<Node> meta_data;
vector<long long> feature_ids;
vector<double> feature_vals;
double* weights;

int total_features;
long epoch;
double learning_rate;
double threshold;


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

__device__ double device_forward(double* weights, long long* feature_ids, double* feature_vals,
               int start, int size) {
    double u = 0;
    for (int i = start; i < start + size; i++) {
        u += feature_vals[i] * weights[feature_ids[i]];
    }
    return device_sigmoid(u);
}

__device__ void update_weights(double* weights, long long* feature_ids, double* feature_vals,
                    double diff, double learning_rate, int start, int size) {
    long long f_idx;
    for (int i = start; i < start + size; i++) {
        f_idx = feature_ids[i];
        atomicAdd(&weights[f_idx], feature_vals[i] * diff * learning_rate);
    }
}

__global__ void train_kernel(double* weights, Node* meta_data, long long* feature_ids, double* feature_vals,
           long epoch, double learning_rate, int meta_data_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < meta_data_size) {
        double y_hat;
        double diff;
        y_hat = device_forward(weights, feature_ids, feature_vals,
                        meta_data[i].data_offset, meta_data[i].num_feature);
        diff = meta_data[i].label - y_hat;
        update_weights(weights, feature_ids, feature_vals,
                diff, learning_rate, meta_data[i].data_offset, meta_data[i].num_feature);
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


int main(int argc, char **argv) {
    string filename = argv[1];
    total_features = stoi(argv[2]);
    epoch = stol(argv[3]);
    learning_rate = stod(argv[4]);


    weights = (double*) malloc(sizeof(double) * total_features);

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

    while (getline(infile, line)) {
        label = line[0] - '0';

        line = line.substr(2);
        std::size_t first, second;
        // cout << "# " << sample_idx << endl;
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
            // cout << " " << f_id <<":"<<f_val;
            if (second == string::npos) break;
            line = line.substr(second);
        }
        // cout << endl;
        // printf("# %d: y=%d, num_feature%d, offset=%d\n", sample_idx, label, num_feature, data_offset);

        Node node = {sample_idx, label, num_feature, data_offset};
        data_offset += num_feature;

        meta_data.push_back(node);
        sample_idx++;
        num_feature = 0;
    }

    // Compute number of blocks and threads per block
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((meta_data.size() + BLOCK_SIZE - 1) / BLOCK_SIZE);

    Node* device_meta_data;
    long long* device_feature_ids;
    double* device_feature_vals;
    double* device_weights;

    // Allocate device memory buffers on the GPU using cudaMalloc
    cudaMalloc((void**)&device_meta_data, meta_data.size() * sizeof(Node));
    cudaMalloc((void**)&device_feature_ids, feature_ids.size() * sizeof(long long));
    cudaMalloc((void**)&device_feature_vals, feature_vals.size() * sizeof(double));
    cudaMalloc((void**)&device_weights, total_features * sizeof(double));

    // Copy sample data into GPU memory
    cudaMemcpy(device_meta_data, &meta_data[0], meta_data.size() * sizeof(Node), cudaMemcpyHostToDevice);
    cudaMemcpy(device_feature_ids, &feature_ids[0], feature_ids.size() * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(device_feature_vals, &feature_vals[0], feature_vals.size() * sizeof(double), cudaMemcpyHostToDevice);

    // run kernel
    for (int e = 0; e < epoch; e++) {
        train_kernel<<<gridDim, blockDim>>>(device_weights, device_meta_data, device_feature_ids, device_feature_vals,
           epoch, learning_rate, meta_data.size());
        cudaDeviceSynchronize();
    }
    // move weights back
    cudaMemcpy(weights, device_weights, total_features * sizeof(double), cudaMemcpyDeviceToHost);
    double error = predict(weights, meta_data, feature_ids, feature_vals);
    printf("error(train): %f\n", error);

    cudaFree(device_meta_data);
    cudaFree(device_feature_ids);
    cudaFree(device_feature_vals);
    cudaFree(device_weights);
    meta_data.clear();
    feature_ids.clear();
    feature_vals.clear();
    free(weights);
    return 0;
}
