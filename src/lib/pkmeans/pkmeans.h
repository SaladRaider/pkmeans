#ifndef __PKMEANS_H_INCLUDED_
#define __PKMEANS_H_INCLUDED_

#include <string>
#include <pkmeans/distribution.h>

namespace pkmeans {
template <class T>
class PKMeans {
    private:
        std::vector<Distribution<T>> clusters;
    public:
        void run (int numClusters, int numBuckets, int numThreads, std::string inFilename,
                  std::string outFilename);
};
}

#endif // __PKMEANS_H_INCLUDED_
