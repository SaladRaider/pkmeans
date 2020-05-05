#ifndef __PKMEANS_H_INCLUDED_
#define __PKMEANS_H_INCLUDED_

#include <string>

namespace pkmeans {
class PKMeans {
    private:
    public:
        void run(int numClusters, int numThreads, std::string inFilename,
                 std::string outFilename);
};
}

#endif // __PKMEANS_H_INCLUDED_
