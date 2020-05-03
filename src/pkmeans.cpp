#include "pkmeans.h"

void PKMeans::run(int numClusters, int numThreads, std::string inFilename,
                  std::string outFilename) {
    printf("pkmeans ran with args (%d, %d, %s, %s)\n",
            numClusters, numThreads, inFilename.c_str(), outFilename.c_str());
}
