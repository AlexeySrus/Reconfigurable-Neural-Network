#include "functional_lib.h"

unsigned long funclib::random(const unsigned long& t1, const unsigned long& t2) {
    return t1 + rand() % (t2 - t1 + 1);
}

void funclib::set_eps(const double& e){
    funclib::GLOB_EPS = e;
}