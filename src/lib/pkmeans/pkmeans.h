#ifndef __PKMEANS_H_INCLUDED_
#define __PKMEANS_H_INCLUDED_

#include <string>
#include <pkmeans/distribution.h>

namespace pkmeans {
class PKMeans {
    private:
        std::vector<Distribution<double>> clusters;
        std::vector<Distribution<double>> distributions;
        std::vector<std::vector<size_t>> clusterAssignments;
        bool converged;

        void readDistributions (std::string inFileName);
        void saveClusters (std::string outFilename);
        void saveAssignments (std::string outFilename);
        void initClusters (int numClusters);
        void clearClusterAssignments ();
        void assignDistributions ();
        void computeNewClusters ();
        void computeClusterMean (size_t idx, Distribution<double> &cluster);
        size_t findClosestCluster (size_t distributionIdx);
    public:
        void run (int numClusters, int numThreads, std::string inFilename,
                  std::string outFilename);
};
}

#endif // __PKMEANS_H_INCLUDED_
