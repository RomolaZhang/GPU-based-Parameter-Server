#include<iostream>
#include<vector>
#include<math.h>
#include<fstream>
#include<string>
#include<chrono>

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

void update_weights(double* weights, vector<long long>& feature_ids, vector<double>& feature_vals,
                    double diff, double learning_rate, int start, int size) {
    long long f_idx;
    for (int i = start; i < start + size; i++) {
        f_idx = feature_ids[i];
        weights[f_idx] += feature_vals[i] * diff * learning_rate;
    }
}

void train(double* weights, vector<Node>& meta_data, vector<long long>& feature_ids, vector<double>& feature_vals,
           long epoch, double learning_rate) {
    int size = meta_data.size();
    double y_hat, diff, gradient;
    for (int e = 0; e < epoch; e++) {
        for (int i = 0; i < size; i++) {
            y_hat = forward(weights, feature_ids, feature_vals,
                            meta_data[i].data_offset, meta_data[i].num_feature);
            diff = meta_data[i].label - y_hat;
            update_weights(weights, feature_ids, feature_vals,
                           diff, learning_rate, meta_data[i].data_offset, meta_data[i].num_feature);
            // printf("epoch=%d, sample=%d, weights=[%f, %f, %f, %f]\n", e, i, weights[0], weights[1], weights[2], weights[3]);
        }
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

    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

    auto start = Clock::now();


    string filename = argv[1];
    total_features = stoi(argv[2]);
    epoch = stol(argv[3]);
    learning_rate = stod(argv[4]);


    weights = (double*) malloc(sizeof(double) * total_features);

    cout << "Input filename: "<< filename << "\n";
    cout << "Numeber of features: "<< total_features << "\n";

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

    auto init_time = duration_cast<dsec>(Clock::now() - start).count();
    printf("Initialization Time: %lf.\n", init_time);

    auto before_train = Clock::now();
    train(weights, meta_data, feature_ids, feature_vals, epoch, learning_rate);
    auto train_time = duration_cast<dsec>(Clock::now() - before_train).count();
    printf("Training Time: %lf.\n", train_time);

    auto before_predict = Clock::now();
    double error = predict(weights, meta_data, feature_ids, feature_vals);
    auto predict_time = duration_cast<dsec>(Clock::now() - before_predict).count();
    printf("Prediction Time: %lf.\n", predict_time);


    printf("error(train): %f\n", error);



  meta_data.clear();
  feature_ids.clear();
  feature_vals.clear();
  free(weights);
  return 0;
}
