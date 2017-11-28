#include "neural_network.hpp"

double a = 1;

double delta = 0.5;

double f(double arg) {
	return 1/(1 + exp(-a*arg));
}

vector<double> f(vector<double> vec){
    for (unsigned int i = 0; i < vec.size(); ++i)
        vec[i] = f(vec[i]);
    return vec;
}

double diff_f(double arg) {
	double f_x = f(arg);
	return a*f_x*(1 - f_x);
}

void set_function_coefficient(double t) {
	if (t < 0)
		return;
	a = t;
}

void set_delta(double t) {
    if (t < 0)
        return;
    delta = t;
}

int random(const int t1, const int t2) {
	return t1 + rand() % t2;
}

double rand_0_1() {
	const int k = 5000;
    int sign = 1;
    if (random(0, 100) > 50)
        sign = -1;
	return ((double)random(0, k)) / k * sign;
}

Neuron::Neuron() {
	local_gradient = 0;
	value = 0;
	in = 0;
	error_code = 0;
	correct_weight = 0;
}

Neuron::Neuron(const unsigned int k) {
	if (!k) {
		error_code = -1;
		return;
	}

	local_gradient = 0;
	value = 0;
	in = k;
	weights.resize(in);
	inputs.assign(in, 0);

	correct_weight = rand_0_1();

	for (unsigned int i = 0; i < in; ++i)
		weights[i] = rand_0_1();

	error_code = 0;
}

Neuron::~Neuron() {
	weights.clear();
	inputs.clear();
}

void Neuron::evaluate() {

	unsigned int i;
	double t = correct_weight;
	for (i = 0; i < in; ++i)
		t += weights[i] * inputs[i];

	value = f(t);
}

double Neuron::get_value() {
	evaluate();
	return value;
}

void Neuron::set_weight(const double value, const unsigned int k) {
	weights[k] = value;
}

double  Neuron::get_weight(const unsigned int k) {
	return weights[k];
}

void Neuron::set_local_gradient(const double v) {
	local_gradient = v;
}

double Neuron::get_local_gradient() {
	return local_gradient;
}

int Neuron::generate_neuron(const unsigned int k) {
	if (!k) {
		error_code = -1;
		return -1;
	}

	local_gradient = 0;
	value = 0;
	in = k;
	weights.resize(in);
	inputs.assign(in, 0);

	correct_weight = rand_0_1();

	for (unsigned int i = 0; i < in; ++i)
		weights[i] = rand_0_1();

	error_code = 0;
	return error_code;
}

void Neuron::input(const vector<double> i) {
	if (i.size() != in) {
		error_code = -2;
		return;
	}
	inputs = i;
}

Neuron Neuron::operator=(const Neuron& right) {
    if (this == &right)
        return *this;

    in = right.in;
    local_gradient = right.local_gradient;
    value = right.value;
    weights = right.weights;
    inputs = right.inputs;
    correct_weight = right.correct_weight;
    error_code = right.error_code;

    return *this;
}

istream& operator>>(istream& is, Neuron& right){
    is >> right.in;
    is >> right.correct_weight;
    right.weights.resize(right.in);
    right.inputs.resize(right.in);
    for (unsigned int i = 0; i < right.in; ++i)
        is >> right.weights[i];
    return is;
}

ostream& operator<<(ostream& os, const Neuron& right){
    os << right.in << ' ' << right.correct_weight << ' ';
    for (auto i : right.weights)
        os << i << ' ';
    return os;
}

void Neuron::initializaton_local_gradient(const double t){
    double v = correct_weight;
	unsigned int i;
    for (i = 0; i < in; ++i)
        v += inputs[i] * weights[i];
    local_gradient = t*diff_f(v);
}

void Neuron::educate() {
    correct_weight += delta*local_gradient;

    for (unsigned int i = 0; i < in; ++i)
        weights[i] += delta*local_gradient*inputs[i];
}

void Neuron::add_synapse(const double w) {
    weights.push_back(w);
    inputs.push_back(0);
    ++in;
}

void Neuron::set_all_weights(const function<double(double, double)>& fill_func) {
    for (double& i : weights)
        i = fill_func(0, 1);
}




NeuralNetwork::NeuralNetwork(unsigned int _input_size, unsigned int layers, const vector<unsigned int>& nns) {
	if (!_input_size) {
		error_code = -2;
		return;
	}

	if (nns.size() != layers) {
		error_code = -1;
		return;
	}

	srand((unsigned int)time(nullptr));

	error_code = 0;
	neurons.resize(layers);
	input_size = _input_size;

	neurons[0].resize(nns[0]);

	for (unsigned int i = 0; i < nns[0]; ++i)
		neurons[0][i].generate_neuron(_input_size);

	for (unsigned int i = 1; i < layers; ++i) {
		neurons[i].resize(nns[i]);
		for (unsigned int j = 0; j < nns[i]; ++j)
			neurons[i][j].generate_neuron(nns[i - 1]);
	}

	output_size = *nns.rbegin();
	output.resize(output_size);


	error_code = 0;
}

NeuralNetwork::NeuralNetwork(unsigned int _input_size, unsigned int layers, const vector<unsigned int>& nns,
              const vector<vector<Neuron> >& ws) {
    if (!_input_size) {
		error_str = "Размер входных данных не соответствует параметрам сети";
        error_code = -2;
        return;
    }

    if (nns.size() != layers) {
		error_str =  "Размер входных данных не соответствует параметрам сети";
        error_code = -1;
        return;
    }

    neurons = ws;

    error_code = 0;
    input_size = _input_size;
    output_size = *nns.rbegin();
    output.resize(output_size);
}

NeuralNetwork::NeuralNetwork(const NeuralNetworkStructure& st) : NeuralNetwork(st.input_size, st.structure.size(), st.structure) {}


NeuralNetwork::~NeuralNetwork() {
	neurons.clear();
}

vector<double> NeuralNetwork::evaluate(const vector<double>& input) {
	vector<double> res(output_size, 0);

	if (input.size() != input_size) {
		error_code = -2;
		error_str = "Размер входных данных не соответствует параметрам сети";
		return res;
	}

	for (unsigned int i = 0; i < neurons[0].size(); ++i)
		neurons[0][i].input(input);

	for (unsigned int i = 1; i < neurons.size(); ++i) {
		vector<double> layer_vec(neurons[i - 1].size());

		for (unsigned int j = 0; j < layer_vec.size(); ++j)
			layer_vec[j] = neurons[i - 1][j].get_value();

		for (unsigned int j = 0; j < neurons[i].size(); ++j)
			neurons[i][j].input(layer_vec);
	}

	for (unsigned int i = 0; i < output_size; ++i)
		res[i] = neurons[neurons.size() - 1][i].get_value();

	return res;
}

NeuralNetwork NeuralNetwork::operator=(const NeuralNetwork& right){
    if (this == &right)
        return *this;

    neurons = right.neurons;
    error_code = right.error_code;
    input_size = right.input_size;
    output_size = right.output_size;
    output = right.output;

    return *this;
}

int NeuralNetwork::get_last_error() {
    return error_code;
}

const vector<vector<Neuron> >& NeuralNetwork::get_neurons_matrix(){
    return neurons;
}

int NeuralNetwork::educate(const vector<double>& _input, const vector<double>& _output) {
    error_code = 0;

	if (_input.size() != input_size) {
        error_code = -1;
		error_str = "Размер входных данных не соответствует параметрам сети";
		return error_code;
	}
	if (_output.size() != output_size) {
        error_code = -2;
		error_str =  "Размер выходных данных не соответствует параметрам сети";
		return error_code;
	}

    vector<double> e(output_size), res = evaluate(_input);

    if (error_code)
        return error_code;


    for (unsigned int i = 0; i < output_size; ++i)
        e[i] = _output[i] - res[i];

    for (unsigned int i = 0; i < neurons[neurons.size() - 1].size(); ++i)
        neurons[neurons.size() - 1][i].initializaton_local_gradient(e[i]);

    for (unsigned int i = 0; i < neurons[neurons.size() - 1].size(); ++i)
        neurons[neurons.size() - 1][i].educate();

    for (int i = (int) neurons.size() - 2; i >= 0; --i)
        for (unsigned int j = 0; j < neurons[i].size(); ++j) {
            double t = 0;
            for (unsigned int k = 0; k < neurons[i + 1].size(); ++k) {
                t += neurons[i + 1][k].get_local_gradient()*neurons[i + 1][k].get_weight(j);
            }
            neurons[i][j].initializaton_local_gradient(t);
            neurons[i][j].educate();
        }

    return 0;
}

string NeuralNetwork::get_last_error_str() { return error_str; }

NeuralNetworkStructure NeuralNetwork::get_structure() const {
    vector<unsigned int> structure(neurons.size());

    for (unsigned long i = 0; i < neurons.size(); ++i)
        structure[i] = static_cast<unsigned int>(neurons[i].size());

    return {input_size, output_size, structure};
}

int NeuralNetwork::add_neuron(unsigned int layer) {
    if (layer >= neurons.size() || !layer){
        error_code = -3;
        error_str = "Некорректный номер слоя для добавления нейрона";
        return error_code;
    }

    neurons[layer - 1].emplace_back(layer == 1 ? input_size : static_cast<unsigned int>(neurons[layer - 2].size()));
    neurons[layer - 1].rbegin()->set_all_weights([](double, double){ return rand_0_1(); });


    for (Neuron& neur : neurons[layer])
        neur.add_synapse(rand_0_1());


    return error_code;
}

int NeuralNetwork::add_layer(unsigned int layer, unsigned int size) {
    if (!size){
        error_code = -4;
        error_str = "Количество нейронов в добавляемом слое не может быть нулевым";
        return error_code;
    }

    if (layer >= neurons.size()){
        error_code = -3;
        error_str = "Некорректный номер слоя для добавления нейрона";
        return error_code;
    }

    neurons.insert(neurons.begin() + layer, vector<Neuron>(size));

    for (Neuron& neur : neurons[layer])
        neur.generate_neuron(!layer ? input_size : static_cast<unsigned int>(neurons[layer - 1].size()));

    for (Neuron& neur : neurons[layer + 1])
        neur.generate_neuron(size);

    return error_code;
}

vector<double> NeuralNetwork::get_layer_output(const vector<double>& input, const unsigned int layer) {
    if (layer >= neurons.size()){
        error_code = -3;
        error_str = "Некорректный номер слоя для вычисления значений слоя";
        return vector<double>();
    }

    vector<double> res(neurons[layer].size(), 0);

    if (input.size() != input_size) {
        error_code = -2;
        error_str = "Размер входных данных не соответствует параметрам сети";
        return res;
    }

    for (unsigned int i = 0; i < neurons[0].size(); ++i)
        neurons[0][i].input(input);

    for (unsigned int i = 1; i < layer + 1; ++i) {
        vector<double> layer_vec(neurons[i -1].size());

        for (unsigned int j = 0; j < layer_vec.size(); ++j)
            layer_vec[j] = neurons[i - 1][j].get_value();

        for (auto& neur : neurons[i])
            neur.input(layer_vec);
    }

    for (unsigned int i = 0; i < neurons[layer].size(); ++i)
        res[i] = neurons[layer][i].get_value();

    return res;
}

vector<vector<pair<double, double> > > analysis_outputs_network_layers(const vector<pair<double, double>>& area,
                                                                        NeuralNetwork& NN){
    NeuralNetworkStructure nn_struct = NN.get_structure();

    vector<vector<pair<double, double> > > res(nn_struct.structure.size());

    for (auto i = 0; i < nn_struct.structure.size(); ++i)
        res[i] = funclib::grad_min_max_of_multidimensional_func<double>(area, [&](const vector<double>& v){
            return NN.get_layer_output(v, static_cast<unsigned int>(i));
        }, 1E-4, 1E-7, 1000);

    return res;
}