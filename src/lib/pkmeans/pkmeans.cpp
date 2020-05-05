#include <string>
#include "pkmeans.h"

using namespace pkmeans;

template <class T>
void PKMeans<T>::run (int numClusters, int numBuckets, int numThreads,
                      std::string inFilename, std::string outFilename) {
    printf ("pkmeans ran with args (%d, %d, %d, %s, %s)\n",
            numClusters, numThreads, numBuckets, inFilename.c_str (),
            outFilename.c_str ());
}

namespace pkmeans {
    template class PKMeans<unsigned int>;
    template class PKMeans<unsigned long long>;
    template class PKMeans<double>;
    template class PKMeans<float>;
}
