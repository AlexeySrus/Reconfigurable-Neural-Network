#pragma once

#include "neural_network.hpp"
#include <string>
#include <iostream>

using namespace std;

class NeuralModel {
private:
    NeuralNetwork NN;
    unsigned int input_size, output_size;
    void refresh_structure();
public:
    vector<unsigned int> structure;
    NeuralModel();
    NeuralModel(unsigned int input_size, const vector<unsigned int>& nns);
    friend istream& operator>>(istream& is, NeuralModel& M);
    friend ostream& operator<<(ostream& os, NeuralModel& M);
    vector<double> evaluate(const vector<double>&);
    int educate(const vector<double>& _input, const vector<double>& _output);
    virtual string get_last_error_str();
    virtual int get_last_error();
    int add_neuron(unsigned int layer);
    int add_layer(unsigned int position, unsigned int size);
    vector<vector<pair<double, double> > > analysis_layers_output(const vector<pair<double, double>>&);
    void flush_neural_network();
    NeuralNetwork get_neural_network();
    void set_neutal_network(const NeuralNetwork&);
};
