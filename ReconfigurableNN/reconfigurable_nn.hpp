#ifndef RECONFIGURABLENN_RECONFIGURABLE_NN_H
#define RECONFIGURABLENN_RECONFIGURABLE_NN_H

#include <iostream>
#include <string>
#include "../neural_network_lib/neural_network.hpp"
#include "../neural_network_lib/NeuralModel.hpp"

inline const double EPS = 1E-5;
#define MAX_ADD_NEURONS 500

using namespace std;

class ReconfigurableNeuralNetwork : public NeuralModel{
private:
    int error_code;
    string error_str;
    double loss;
    vector<vector<double>> input_data, output_data;
    unsigned int input_size;
    bool data_availability;
public:
    ReconfigurableNeuralNetwork();
    ReconfigurableNeuralNetwork(unsigned int, unsigned int);
    ReconfigurableNeuralNetwork(const vector<vector<double>>& inp, const vector<vector<double>>& out);
    ReconfigurableNeuralNetwork(const unsigned int max_layer_count,
            const vector<vector<double>>& inp, const vector<vector<double>>& out);
    ~ReconfigurableNeuralNetwork();
    int get_last_error() override;
    string get_last_error_str() override;
    int fit(function<bool(const unsigned long, const unsigned long,const unsigned long, const unsigned long)>, const double _loss);
};

unsigned long max_index_from_vector(const vector<double>&);

template <unsigned int coeff>
bool DEFAULT_TRAPEZE(const unsigned long inp_size, const unsigned long out_size,
                     const unsigned long size, const unsigned long layer){
    long value = coeff*inp_size - layer;
    if (value < 0)
        return size <= out_size;
    return ((value > out_size && size < value) || size <= out_size);
}

#endif //RECONFIGURABLENN_RECONFIGURABLE_NN_H
