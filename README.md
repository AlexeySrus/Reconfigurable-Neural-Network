# Reconfigurable Neural Network
Реализация реконфигурируемой нейронной сети и метода анализа выходов нейронов
---
### Описание библеотек
    /neural_network_lib/neural_network.hpp
Содержит реализацию нейронной сети с обучением, основанном на алгоритме обратного распространения ошибки. В коде лучше и удобнее будет использовать следующую библеотеку. Из данной библеотки могут понадобится следующие функции и константы:
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
