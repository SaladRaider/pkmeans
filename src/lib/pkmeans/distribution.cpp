#include "distribution.h"

namespace pkmeans {
template struct Distribution<unsigned int>;
template struct Distribution<unsigned long long>;
template struct Distribution<double>;
template struct Distribution<float>;
}  // namespace pkmeans
