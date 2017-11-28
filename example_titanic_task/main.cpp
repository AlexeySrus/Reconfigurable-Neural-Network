#include <iostream>
#include <vector>
#include <string>
#include "../ReconfigurableNN/reconfigurable_nn.hpp"
#include <fstream>
#include <set>
#include <cmath>

using namespace std;

vector<string> line_processing(const string& str, const char& delimiter){
    vector<string> res;
    unsigned long prev_position = 0, next_position, symb_shift;

    while ((next_position = str.find(delimiter, prev_position)) != string::npos) {
        symb_shift = 0;

        if (str[prev_position] == '\"') {
            next_position = str.find(string("\"") + delimiter, next_position) + 1;
            symb_shift = 0;
        }

        res.emplace_back(str.begin() + prev_position + symb_shift, str.begin() + next_position - symb_shift);

        prev_position = next_position + 1;
    }

    res.emplace_back(str.begin() + prev_position, str.begin() + str.length() - 1);

    return res;
}

template <typename T>
bool in_set_insert(set<T>& in_set, const T& value){
    if (in_set.find(value) != in_set.end())
        return true;
    in_set.emplace(value);
    return false;
}

template <typename T>
bool in_set(set<T>& in_set, const T& value){
    return (in_set.find(value) != in_set.end());
}

double softmax(const vector<double>& v, const unsigned long& k){
    double d = 0;
    for (const auto& i : v)
        d += std::exp(i);
    return exp(v[k]) / d;
}

double avg_positive(const vector<double>& v){
    double res = 0;
    unsigned long count = 0;

    for (auto& i : v)
        if (i > 0) {
            res += i;
            ++count;
        }

    return res / count;
}

int main(int argc, char** argv) {

    if (argc < 2){
        cout << "Переданно недостаточное количество аргументов командной строки" << endl;
        return EXIT_FAILURE;
    }

    ifstream inp_file(argv[1]);

    if (!inp_file){
        cout << "Не удалось открыть файл: " << argv[1] << endl;
        return EXIT_FAILURE;
    }

    vector<string> file_strings;

    for (char tmp_str[128]; inp_file.getline(tmp_str, 128); file_strings.emplace_back(tmp_str));

    inp_file.close();

    string title = *file_strings.begin();

    file_strings.erase(file_strings.begin());

    vector<vector<string>> table_in_strs;

    for (const auto& i : file_strings)
        table_in_strs.push_back(line_processing(i, ','));

    file_strings.clear();


    auto n_colms = table_in_strs.begin()->size();
    vector<set<string>> columns_values(n_colms);

    for (const auto& strs : table_in_strs)
        for (auto j = 0; j < n_colms; ++j)
            in_set_insert(columns_values[j], strs[j]);


    for (auto i = 0; i < n_colms; ++i)
        if (in_set(columns_values[i], string("")))
            cout << "В колонке " << i << " есть пустые значения" << endl;

    vector<vector<double> > table_in_double(table_in_strs.size());
    vector<vector<double> > results(table_in_strs.size());


    for (auto i = 0; i < table_in_strs.size(); ++i)
        if (table_in_strs[i][1] == "0")
            results[i] = { TRUE_VALUE, FALSE_VALUE };
        else
            results[i] = { FALSE_VALUE, TRUE_VALUE };

    for (auto i = 0; i < table_in_strs.size(); ++i){
        table_in_double[i].push_back(stod(table_in_strs[i][2]));
        table_in_double[i].push_back((table_in_strs[i][4] == "male" ? 1 : 0));
        table_in_double[i].push_back((table_in_strs[i][5].empty() ? -1 : stod(table_in_strs[i][5])));
        table_in_double[i].push_back(stod(table_in_strs[i][6]));
        table_in_double[i].push_back(stod(table_in_strs[i][7]));
        table_in_double[i].push_back((table_in_strs[i][9].empty() ? -1 : stod(table_in_strs[i][9])));

        if (table_in_strs[i][11].empty())
            table_in_double[i].push_back(0);
        else{
            if (table_in_strs[i][11] == "C")
                table_in_double[i].push_back(0);

            if (table_in_strs[i][11] == "Q")
                table_in_double[i].push_back(1);

            if (table_in_strs[i][11] == "S")
                table_in_double[i].push_back(2);
        }

        const double k = 10;

        if (table_in_strs[i][3].find("Mr.", 0) != string::npos)
            table_in_double[i].push_back(1/k);
        else
            if (table_in_strs[i][3].find("Mrs.", 0) != string::npos)
                table_in_double[i].push_back(2/k);
            else
                if (table_in_strs[i][3].find("Miss.", 0) != string::npos)
                    table_in_double[i].push_back(3/k);
                else
                    if (table_in_strs[i][3].find("Don.", 0) != string::npos)
                        table_in_double[i].push_back(4/k);
                    else
                        if (table_in_strs[i][3].find("Master.", 0) != string::npos)
                            table_in_double[i].push_back(5/k);
                        else
                            table_in_double[i].push_back(0);
    }

    table_in_strs.clear();


    vector<vector<double>> columns_double(table_in_double.begin()->size());
    vector<double> columns_avg(table_in_double.begin()->size());

    for (auto j = 0; j < table_in_double.begin()->size(); ++j) {
        for (auto i = 0; i < table_in_double.size(); ++i)
            columns_double[j].push_back(table_in_double[i][j]);
        columns_avg[j] = avg_positive(columns_double[j]);
    }

    for (auto j = 0; j < table_in_double.begin()->size(); ++j)
        for (auto i = 0; i < table_in_double.size(); ++i)
            if (table_in_double[i][j] < 0) {
                columns_double[j][i] = columns_avg[j];
                table_in_double[i][j] = columns_avg[j];
            }

    for (auto i = 0; i < table_in_double.size(); ++i)
        for (auto j = 0; j < table_in_double.begin()->size(); ++j)
            if (j != 1 && j != 2 && j != 7) {
                table_in_double[i][j] = (table_in_double[i][j] / columns_avg[j]);
                columns_double[j][i] = table_in_double[i][j];
            }

    auto max_age = funclib::max(columns_double[2], funclib::GLOB_EPS);
    for (auto i = 0; i < table_in_double.size(); ++i) {
        table_in_double[i][2] /= max_age / 2;
        columns_double[2][i] = table_in_double[i][2];
    }

    vector<pair<double, double>> min_max_column_values;
    for(auto i = 0; i < columns_double.size(); ++i)
        min_max_column_values.emplace_back(funclib::min(columns_double[i], funclib::GLOB_EPS),
                                           funclib::max(columns_double[i], funclib::GLOB_EPS));





    ReconfigurableNeuralNetwork NeurNet(table_in_double, results);
    NeurNet.fit(DEFAULT_TRAPEZE<40>, 0.18);

    auto analysis_res = NeurNet.analysis_layers_output(min_max_column_values);

    for (auto i = 0; i < analysis_res.size(); ++i){
        for (auto j = 0; j < ((analysis_res[i].size() > 10 ) ? 10 : analysis_res.size()); ++j)
            cout << "[" << analysis_res[i][j].first << ", " << analysis_res[i][j].second << "] ";
        cout << endl;
    }



    return EXIT_SUCCESS;
}