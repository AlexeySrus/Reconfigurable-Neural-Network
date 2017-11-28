#pragma once

#include <cmath>
#include <ctime>
#include <vector>
#include <iostream>
#include <random>
#include <functional>
#include "../FuncAnalyze/functional_lib.h"

using namespace std;

double f(double);

vector<double> f(vector<double>);

double diff_f(double);

void set_function_coefficient(double);

void set_delta(double);

double rand_0_1();

const double TRUE_VALUE = 1000.0;
const double FALSE_VALUE = -1000.0;

struct NeuralNetworkStructure{
    unsigned int input_size;
    unsigned int output_size;
    vector<unsigned int> structure;
};

class Neuron {
private:
	unsigned int in;
	double local_gradient;
	double value;
	vector<double> weights;
	vector<double> inputs;
	double correct_weight;
	void evaluate();
	int error_code;
public:
	Neuron();
	Neuron(const unsigned int);
	int generate_neuron(const unsigned int);
	double get_value();
	void set_weight(const double, const unsigned int);
	double get_weight(const unsigned int);
	~Neuron();
	void set_local_gradient(const double);
	double get_local_gradient();
    void initializaton_local_gradient(const double);
	void input(const vector<double>);
    Neuron operator=(const Neuron& right);
    friend istream& operator>>(istream& is, Neuron& right);
    friend ostream& operator<<(ostream& os, const Neuron& right);
    void educate();
    void add_synapse(const double);
    void set_all_weights(const function<double(double, double)>&);
};

class NeuralNetwork {
private:
	vector<vector<Neuron> > neurons;
	string error_str;
	int error_code;
	unsigned int input_size, output_size;
	vector<double> output;
public:
	//Размерность входа, Количество слоёв, и Количество нейронов в каждом слое
	NeuralNetwork(unsigned int _input_size, unsigned int layers, const vector<unsigned int>& nns);
    NeuralNetwork(unsigned int, unsigned int, const vector<unsigned int>&, const vector<vector<Neuron> >&);
    NeuralNetwork(const NeuralNetworkStructure&);
	~NeuralNetwork();
	vector<double> evaluate(const vector<double>& input);
    NeuralNetwork operator=(const NeuralNetwork& right);
    int get_last_error();
    const vector<vector<Neuron> >& get_neurons_matrix();
    int educate(const vector<double>& _input, const vector<double>& _output);
	string get_last_error_str();
    int add_neuron(unsigned int layer);
    int add_layer(unsigned int layer, unsigned int size);
    NeuralNetworkStructure get_structure() const;
	vector<double> get_layer_output(const vector<double>& input, const unsigned int layer);
};

vector<vector<pair<double, double> > > analysis_outputs_network_layers(const vector<pair<double, double>>&,
                                                                        NeuralNetwork&);
