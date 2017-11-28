#include "NeuralModel.hpp"

NeuralModel::NeuralModel() : NN(1, 1, {1}) {}

NeuralModel::NeuralModel(unsigned int _input_size, const vector<unsigned int> &nns) :
    NN(_input_size, nns.size(), nns) {
    structure = nns;
    input_size = _input_size;
    output_size = *nns.rbegin();
}

istream& operator>>(istream& is, NeuralModel& M){
    is >> M.input_size;
    unsigned int layers;
    is >> layers;
    M.structure.resize(layers);
    for (unsigned int i = 0; i < layers; ++i)
        is >> M.structure[i];

    vector<vector<Neuron> > ws(layers);
    for (unsigned int i = 0; i < layers; ++i) {
        ws[i].resize(M.structure[i]);
        for (unsigned int j = 0; j < M.structure[i]; ++j)
            is >> ws[i][j];
    }

    M.output_size = *M.structure.rbegin();

    M.NN = NeuralNetwork(M.input_size, layers, M.structure, ws);
    return is;
}

ostream& operator<<(ostream& os, NeuralModel& M){
    os << M.input_size << ' ' << M.structure.size() << ' ';
    for (auto i : M.structure)
        os << i << ' ';
    for (auto i : M.NN.get_neurons_matrix()){
        for (auto k : i)
            os << k << ' ';
        os << endl;
    }
    return os;
}

vector<double> NeuralModel::evaluate(const vector<double>& input) {
    return NN.evaluate(input);
}

int NeuralModel::educate(const vector<double>& _input, const vector<double>& _output){
    return NN.educate(_input, f(_output));
}

string NeuralModel::get_last_error_str() { return NN.get_last_error_str(); }

int NeuralModel::get_last_error() { return NN.get_last_error(); }

void NeuralModel::refresh_structure() {
    NeuralNetworkStructure S = NN.get_structure();
    input_size = S.input_size;
    output_size = S.output_size;
    structure = S.structure;
}

int NeuralModel::add_neuron(unsigned int layer){
    int err = NN.add_neuron(layer);
    if (!err)
        refresh_structure();
    return err;
}

int NeuralModel::add_layer(unsigned int position, unsigned int size){
    int err = NN.add_layer(position, size);
    if (!err)
        refresh_structure();
    return err;
}

vector<vector<pair<double, double> > > NeuralModel::analysis_layers_output(const vector<pair<double, double>>& area){
    return analysis_outputs_network_layers(area, NN);
}

void NeuralModel::flush_neural_network(){
    NN = NeuralNetwork(NN.get_structure());
}

NeuralNetwork NeuralModel::get_neural_network(){
    return NN;
}

void NeuralModel::set_neutal_network(const NeuralNetwork& NN){
    this->NN = NN;
    refresh_structure();
}