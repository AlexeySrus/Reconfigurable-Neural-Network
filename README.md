# Reconfigurable Neural Network
Реализация реконфигурируемой нейронной сети и метода анализа выходов нейронов
---
### Краткое описание библеотек
---
    /neural_network_lib/neural_network.hpp
Содержит реализацию нейронной сети с обучением, основанном на алгоритме обратного распространения ошибки и метод анализа выходов нейронов. В коде лучше и удобнее будет использовать следующую по списку библеотеку. Из данной библеотки могут понадобится следующие функции и константы:
```c++
//Изменение коэффициента при экспоненте в сигмоидальной функции активации
//По умолчанию 1
void set_function_coefficient(double);

//Изменение коэффициента для градиентного спуска
//По умолчанию 0.5
void set_delta(double);

//Константы, при которых функция активации стремится к своему максимуму и минимуму
//В данном случае f(x) in 0..1
const double TRUE_VALUE = 1000.0;
const double FALSE_VALUE = -1000.0;
```
---
     /neural_network_lib/NeuralModel.hpp
Содержит более удобный в использовании класс `NeuralModel`.
##### Пример использования конструктора класса `NeuralModel`
```c++
/*
    Создасётся нейронная сеть с двумя скрытыми стоями по 25 и 12 нейронов,
    принимающая на входе вектор из 12 значений и возвращающая
    вектор из двух значений
    (последний элемент вектора структуры НС характеризует выходной слой)
*/
NeuralModel Model(12, {25, 12, 2});

//Создаётся нейронная сеть с одним выходым слоем, принимающая один входной элемент
NeuralModel Model2();
```
##### Обучение `NeuralModel`
```c++
//Заполнение обучающей выбоки, размер каждого вектора должен совпадать с размерностью входа НС
vector<vector<double> > train_X(N);
vector<vector<double> > train_Y(N);

for (auto& i : train_X)
	i.assign(12, ...);	
	
for (auto& i : train_Y)
	i.assign(2, ...);

//Процесс обучения
for (auto i = 0; i < N; ++i) {
	int err = Model.educate(train_X[i], train_Y[i]);
	if (err) {
		//Обработка ошибки
		//В случае успешного завершения все методы возвращают 0 (EXIT_SUCCESS)
		.....
	}
}
```
##### Работа `NeuralModel`
```c++
vector<double> input(12);
vector<double> output(2);

int err = Model.educate(input, output);
```
Стоит отметить, что в классах `NeuralNetwork` и `NeuralModel` есть методы, возвращающие код коследней ошибки и строку типа `std::string` с её кратким описанием.
```c++
int err = Model.get_last_error();
string err_string = Model.get_last_error_str();
```
Для класса `NeuralModel` перегруженны операции записи и чтения с потока.
```c++
//Запись в файл
ofstream file("neualnetwork.nn");
file << Model;
file.close();

//Чтение с файла
ifstream nn_file("neualnetwork.nn");
nn_file >> Model;
file.close();
```
##### Пример использования метода анализа нейронной сети `NeuralModel::analysis_layers_output`
```c++
vector<vector<pair<double, double> > > analysis_res = Model.analysis_layers_output(nn_inputs);

for (auto i = 0; i < analysis_res.size(); ++i){
    for (auto j = 0; j < analysis_res[i].size(); ++j)
        cout << "[" << analysis_res[i][j].first << ", " << analysis_res[i][j].second << "] ";
    cout << endl;
}
```
---
	/ReconfigurableNN/NeuralModel.hpp
Содержит реализацию самореконфигурируемой нейронной сети. Класс `ReconfigurableNeuralNetwork` наследуется от класса `NeuralModel`, поэтому методы обучения и работы остаются теми же.
Конструктору данной нейронной сети требуется передать лишь матрицу входных данных для обучения с ответами:
```c++
ReconfigurableNeuralNetwork RNN(train_X, train_Y);
```
После чего вызвать метод "тренировки" сети `ReconfigurableNeuralNetwork::fit`, принимающий функцию, определяющюю структуру сети и желаемую ошибку сети на тестовых данных. В случае, если данная ошибка не была достигнута, нейронная сеть вернётся в "наилучшее" состояние (в состояние с наименьшей ошибкой). Так же на каждом шаге в поток `std::cout` будет выводиться изменения структуры сети, текущая ошибка и минимальная ошибка, которой удалось достичь.
```c++
int err = RNN.fit(DEFAULT_TRAPEZE<5>, 0.15);
```
