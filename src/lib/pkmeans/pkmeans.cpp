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

template void PKMeans<unsigned int>::run (
    int numClusters, int numBuckets, int numThreads, std::string inFilename,
    std::string outFilename);

template void PKMeans<unsigned long long>::run (
    int numClusters, int numBuckets, int numThreads, std::string inFilename,
    std::string outFilename);

template void PKMeans<double>::run (
    int numClusters, int numBuckets, int numThreads, std::string inFilename,
    std::string outFilename);

template void PKMeans<float>::run (
    int numClusters, int numBuckets, int numThreads, std::string inFilename,
    std::string outFilename);
