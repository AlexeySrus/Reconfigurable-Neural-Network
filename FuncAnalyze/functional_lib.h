#ifndef FINDMINMAXFUNC_FUNCTIONAL_LIB_H
#define FINDMINMAXFUNC_FUNCTIONAL_LIB_H

#include <functional>
#include <vector>
#include <cstdlib>

namespace funclib {

    const long long MAX_STEPS = 10000;

    inline double GLOB_EPS = 1E-5;

    unsigned long random(const unsigned long &t1, const unsigned long &t2);

    void set_eps(const double &);

    template<typename T>
    T abs(const T &arg) { return arg < 0 ? -arg : arg; }

    template<typename T>
    std::vector<T> set_of_points(const T &left, const T &right, const unsigned long &count) {
        if (count == 1)
            return std::vector<T>({(left + right) / 2});

        std::vector<T> res_v(count);
        for (auto i = 0; i < count; ++i)
            res_v[i] = left + i * (right - left) / (count - 1);

        return res_v;
    }

    template<typename T>
    T max(const std::vector<T> &v, const T &EPS) {
        T max_elem = *v.begin();
        for (auto &i : v)
            if (i - max_elem > -EPS)
                max_elem = i;
        return max_elem;
    }

    template<typename T>
    T min(const std::vector<T> &v, const T &EPS) {
        T min_elem = *v.begin();
        for (auto &i : v)
            if (i - min_elem < EPS)
                min_elem = i;
        return min_elem;
    }

    template<typename T>
    bool in(const std::vector<T> &v, const T &elem, const T &EPS) {
        for (auto &i : v)
            if (abs(i - elem) < EPS)
                return true;
        return false;
    }


    template<typename T>
    T grad_max_of_func(const std::vector<std::pair<T, T>> &args_area,
                       std::function<T(const std::vector<T> &)> explored_func,
                       const T &h, const T &lambda, const std::vector<T> &start_point, T EPS) {
        using namespace std;

        auto n = args_area.size();

        auto vec_add = [](vector<T> x, long i, T value) {
            x[i] += value;
            return x;
        };

        auto gradient = [&](const vector<T> &x) {
            vector<T> grad_values;
            for (auto i = 0; i < n; ++i)
                grad_values.emplace_back((-3 * explored_func(x) - 4 * explored_func(vec_add(x, i, h)) -
                                          explored_func(vec_add(x, i, 2 * h))) / 2 / h);
            return grad_values;
        };


        auto rate = [](const vector<T> &x) {
            T res = 0, s = 1;
            for (auto &i : x)
                res += i * i;

            for (auto i = 0; i < 500; ++i)
                s = (s + res / s) / 2;

            return s;
        };

        auto add_vec_mul_coeff = [](vector<T> &v_res, const vector<T> &v_add, const T &a) {
            for (auto i = 0; i < v_res.size(); ++i)
                v_res[i] += v_add[i] * a;
            return;
        };

        auto add_vec_mul_coeff_res = [](vector<T> v_res, const vector<T> &v_add, const T &a) {
            for (auto i = 0; i < v_res.size(); ++i)
                v_res[i] += v_add[i] * a;
            return v_res;
        };

        const unsigned long count_protection_points = 2, max_count_equals_loop_points = 5;
        vector<T> loop_protection_vec(count_protection_points);
        unsigned long loop_protection_id = 0;
        bool loop_protection_start = false;
        long long explicit_stop = 0;

        vector<T> x_next = start_point, x_pred = x_next;
        T fn = explored_func(x_pred) + 10 * EPS, tmp = 0;

        while (abs(explored_func(x_next) - fn) >= EPS) {
            x_pred = x_next;
            fn = explored_func(x_pred);

            auto g = gradient(x_next);
            T rate_g = rate(g);

            add_vec_mul_coeff(x_next, g, -lambda / rate_g);

            for (auto i = 0; i < x_next.size(); ++i)
                if (x_next[i] - args_area[i].first < EPS)
                    x_next[i] = args_area[i].first;
                else if (x_next[i] - args_area[i].second > -EPS)
                    x_next[i] = args_area[i].second;

            loop_protection_id = ++loop_protection_id % count_protection_points;
            loop_protection_vec[loop_protection_id] = fn;
            if (!loop_protection_start && !loop_protection_id)
                loop_protection_start = true;
            else if (loop_protection_start)
                if (in(loop_protection_vec, tmp = explored_func(x_next), EPS))
                    loop_protection_vec.push_back(tmp);
            if (loop_protection_vec.size() > max_count_equals_loop_points + count_protection_points)
                return max(loop_protection_vec, EPS);

            if (++explicit_stop > MAX_STEPS) {
                EPS *= 10;
                explicit_stop = 0;
            }
        }

        return max(vector<T>({explored_func(x_next), fn}), EPS);
    }

    template<typename T>
    std::pair<T, T> grad_min_max_of_func(const std::vector<std::pair<T, T>> &args_area,
                                         std::function<T(const std::vector<T> &)> explored_func,
                                         const T &h, const T &lambda, const unsigned long &point_count) {
        using namespace std;

        srand(static_cast<unsigned int>(time(nullptr)));

        auto n = args_area.size();

        vector<vector<T>> v_points;

        vector<T> left_p, right_p;

        for (auto &i : args_area) {
            v_points.push_back(set_of_points(i.first, i.second, point_count));
            left_p.push_back(i.first);
            right_p.push_back(i.second);
        }

        vector<T> v_res;

        v_res.push_back(explored_func(left_p));
        v_res.push_back(explored_func(right_p));

        for (auto i = 0; i < point_count; ++i) {
            vector<T> x;
            for (auto &p : v_points)
                x.push_back(p[random(0, p.size() - 1)]);

            T t_eps = GLOB_EPS;

            v_res.push_back(
                    -grad_max_of_func<T>(args_area, [&explored_func](const vector<T> &v) { return -explored_func(v); },
                                         h, lambda, x, t_eps));

            v_res.push_back(grad_max_of_func<T>(args_area, explored_func, h, lambda, x, t_eps));
        }

        return make_pair(min(v_res, T(GLOB_EPS)), max(v_res, T(GLOB_EPS)));
    }

    template<typename T>
    std::vector<std::pair<T, T>> grad_min_max_of_multidimensional_func(const std::vector<std::pair<T, T>> &args_area,
                                                                       std::function<std::vector<T>(
                                                                               const std::vector<T> &)> explored_func,
                                                                       const T &h, const T &lambda,
                                                                       const unsigned long &point_count) {
        using namespace std;
        vector<pair<T, T>> res;

        vector<T> test_func_input;

        for (auto &i :args_area)
            test_func_input.push_back((i.first + i.second) / 2);

        auto func_out_dim = explored_func(test_func_input).size();
        for (auto i = 0; i < func_out_dim; ++i)
            res.push_back(grad_min_max_of_func<T>(args_area, [&](const vector<T> &v) {
                return explored_func(v)[i];
            }, h, lambda, point_count));

        return res;
    }
}

#endif //FINDMINMAXFUNC_FUNCTIONAL_LIB_H
