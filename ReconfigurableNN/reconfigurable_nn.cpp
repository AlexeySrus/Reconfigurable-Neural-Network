#include "reconfigurable_nn.hpp"

using namespace std;

ReconfigurableNeuralNetwork::ReconfigurableNeuralNetwork() : NeuralModel(), error_code(0), error_str(""), data_availability(false) ,
    input_size(1) {};

ReconfigurableNeuralNetwork::ReconfigurableNeuralNetwork(unsigned int _input_size, unsigned int _output_size) :
        NeuralModel(_input_size, {_output_size}), error_code(0), error_str(""), data_availability(false), input_size(_input_size) {};

ReconfigurableNeuralNetwork::ReconfigurableNeuralNetwork(const vector<vector<double>>& inp, const vector<vector<double>>& out):
    NeuralModel(static_cast<unsigned int>(inp[0].size()), {static_cast<unsigned int>(out[0].size())}) {
    data_availability = false;

    for (const auto& i : inp)
        if (i.size() != inp[0].size()){
            error_code = -1;
            error_str = "Входной набор данных имеет неодинаковые размерности";
            return;
        }

    for (const auto& i : out)
    if (i.size() != out[0].size()){
        error_code = -1;
        error_str = "Входной набор данных обучения имеет неодинаковые размерности";
        return;
    }

    if (inp.size() != out.size()){
        error_code = -1;
        error_str = "Количество элемнтов в входном наборе не соответствует размеру набора обучения";
        return;
    }

    input_data = inp;
    output_data = out;
    data_availability = true;

    input_size = static_cast<unsigned int>(inp[0].size());
}

ReconfigurableNeuralNetwork::ReconfigurableNeuralNetwork(const unsigned int max_layer_count,
                            const vector<vector<double>>& inp, const vector<vector<double>>& out):
        ReconfigurableNeuralNetwork(inp, out) {

    auto t = static_cast<int>(max_layer_count - structure.size());

    for (auto i = 0; i < t; ++i)
        add_layer(0, 1);
}

ReconfigurableNeuralNetwork::~ReconfigurableNeuralNetwork(){
    input_data.clear();
    output_data.clear();
}

int ReconfigurableNeuralNetwork::get_last_error() {
    int base_err = NeuralModel::get_last_error();
    return (base_err ? base_err : error_code);
}

string ReconfigurableNeuralNetwork::get_last_error_str() {
    return (NeuralModel::get_last_error() ? NeuralModel::get_last_error_str() : error_str);
}

int ReconfigurableNeuralNetwork::fit(
        function<bool(const unsigned long, const unsigned long,const unsigned long, const unsigned long)> check_f,
        const double _loss) {
    if (!data_availability){
        error_code = -4;
        error_str = "Попытка вызвать обучение модели без загруженных данных";
        return error_code;
    }

    loss = _loss;
    NeuralNetwork best_NN = this->get_neural_network();

    unsigned long n = input_data.size();

    for (unsigned long i = 0; i < n; ++i){
        error_code = NeuralModel::educate(input_data[i], output_data[i]);
        if (error_code)
            return error_code;
    }

    vector<unsigned long> err_elements;
    unsigned long num_add_neurons = 0;

    for (unsigned long i = 0; i < n; ++i)
        if (max_index_from_vector(NeuralModel::evaluate(input_data[i])) != max_index_from_vector(output_data[i]))
            err_elements.push_back(i);

    double now_loss = static_cast<double>(err_elements.size()) / n, min_loss = now_loss;
    cout << "Start loss: " << now_loss << endl;

    while (num_add_neurons <= MAX_ADD_NEURONS) {

        if (now_loss - loss < EPS){
            best_NN = this->get_neural_network();
            min_loss = now_loss;
            break;
        }

        if (now_loss - min_loss < EPS) {
            min_loss = now_loss;
            best_NN = this->get_neural_network();
        }

        cout << "loss: " << now_loss << ", min loss: " << min_loss << endl;

        bool add_neuron = false, add_layer = false;
        unsigned long layer = 0;

        while (!(add_neuron || add_layer)){

            if (check_f(input_size, *NeuralModel::structure.rbegin(), NeuralModel::structure[layer], layer) &&
                    layer != structure.size() - 1){
                error_code = NeuralModel::add_neuron(static_cast<unsigned int>(layer + 1));

                if (error_code)
                    return error_code;

                add_neuron = true;

            } else {

                ++layer;

                if (layer == structure.size()) {
                    error_code = NeuralModel::add_layer(static_cast<unsigned int>(layer - 1), 1);

                    if (error_code)
                        return error_code;

                    add_layer = true;
                }
            }
        }

        if (add_neuron){

            for (unsigned long i = 0; i < n; ++i){
                error_code = NeuralModel::educate(input_data[i], output_data[i]);
                if (error_code)
                    return error_code;
            }

            err_elements.clear();

            for (unsigned long i = 0; i < n; ++i)
                if (max_index_from_vector(NeuralModel::evaluate(input_data[i])) != max_index_from_vector(output_data[i]))
                    err_elements.push_back(i);

            now_loss = static_cast<double>(err_elements.size()) / n;

        }else{

            for (unsigned long i = 0; i < n; ++i){
                error_code = NeuralModel::educate(input_data[i], output_data[i]);
                if (error_code)
                    return error_code;
            }

            err_elements.clear();

            for (unsigned long i = 0; i < n; ++i)
            if (max_index_from_vector(NeuralModel::evaluate(input_data[i])) != max_index_from_vector(output_data[i]))
                err_elements.push_back(i);

            now_loss = static_cast<double>(err_elements.size()) / n;
        }

        cout << "Changed structude: " << endl;
        for (const auto i : NeuralModel::structure)
            cout << i << ' ';
        cout << endl;

        ++num_add_neurons;
    }

    this->set_neutal_network(best_NN);

    cout << endl << "result loss: " << min_loss << endl;
    cout << "result structure: ";
    for (const auto i : NeuralModel::structure)
        cout << i << ' ';
    cout << endl;

    return error_code;
}

unsigned long max_index_from_vector(const vector<double>& v){
    unsigned long max_index = 0;
    double max_elem = 0;

    for (unsigned long i = 0; i < v.size(); ++i)
        if (v[i] - max_elem > -EPS){
            max_elem = v[i];
            max_index = i;
        }

    return max_index;
}