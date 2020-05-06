#ifndef __PKMEANS_H_INCLUDED_
#define __PKMEANS_H_INCLUDED_

#include <gtest/gtest.h>
#include <pkmeans/distribution.h>
#include <string>

namespace pkmeans {
class PKMeans {
    private:
        std::vector<Distribution<double>> clusters;
        std::vector<Distribution<double>> distributions;
        std::vector<std::vector<size_t>> clusterAssignments;
        std::vector<size_t> prevClusterAssignments;
        bool converged = false;

        void readDistributions (std::string inFileName);
        void saveClusters (std::string outFilename);
        void saveAssignments (std::string outFilename);
        void initClusters (int numClusters);
        void clearClusterAssignments ();
        void assignDistributions ();
        void computeNewClusters ();
        void computeClusterMean (size_t idx, Distribution<double> &cluster);
        size_t findClosestCluster (size_t distributionIdx);

        // test friend functions
        friend class PKMeansTests;
        FRIEND_TEST (PKMeansTests, ReadDistributions);
        FRIEND_TEST (PKMeansTests, InitClusters);
        FRIEND_TEST (PKMeansTests, FindClosestCluster);
        FRIEND_TEST (PKMeansTests, AssignDistributions);
        FRIEND_TEST (PKMeansTests, ClearClusterAssignments);
        FRIEND_TEST (PKMeansTests, ComputeClusterMean);
        FRIEND_TEST (PKMeansTests, ComputeNewClusters);
    public:
        void run (int numClusters, int numThreads, std::string inFilename,
                  std::string assignmentsOutFilename, std::string clustersOutFilename);
};
}

#endif // __PKMEANS_H_INCLUDED_
