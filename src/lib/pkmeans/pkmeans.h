#ifndef __PKMEANS_H_INCLUDED_
#define __PKMEANS_H_INCLUDED_

#include <gtest/gtest.h>
#include <pkmeans/distribution.h>
#include <string>

namespace pkmeans {
class PKMeans {
    private:
        std::vector<Distribution<double>> clusters;
        std::vector<Distribution<double>> newClusters;
        std::vector<Distribution<double>> distributions;
        std::vector<std::vector<size_t>> clusterAssignments;
        std::vector<size_t> prevClusterAssignments;
        std::vector<std::vector<double>> lowerBounds;
        std::vector<double> upperBounds;
        std::vector<std::vector<double>> clusterDists;
        std::vector<std::vector<double>> clusterDistributionDists;
        std::vector<double> sDists;
        std::vector<double> r;
        bool converged = false;

        void readDistributions (std::string inFileName);
        void saveClusters (std::string outFilename);
        void saveAssignments (std::string outFilename);
        void initClusters (int numClusters);
        void initNewClusters ();
        void initLowerBounds ();
        void initUpperBounds ();
        void initAssignments ();
        void initClusterDistributionDists ();
        void initClusterDists ();
        void initSDists ();
        void initR ();
        void computeClusterDists ();
        void computeSDists ();
        void resetR ();
        void clearClusterAssignments ();
        void assignDistributions ();
        void assignNewClusters ();
        void computeNewClusters ();
        void computeLowerBounds ();
        void computeUpperBounds ();
        void computeClusterMean (size_t idx);
        double calcObjFn ();
        bool needsClusterUpdate (size_t x, size_t c);
        size_t findClosestCluster (size_t distributionIdx);
        size_t findClosestInitCluster (size_t distributionIdx);
        size_t getCluster (size_t distributionIdx);
        double computeDcDist (size_t x, size_t c);
        double dcDist (size_t x, size_t c);
        double cDist (size_t c1, size_t c2);

        // test friend functions
        friend class PKMeansTests;
        FRIEND_TEST (PKMeansTests, ReadDistributions);
        FRIEND_TEST (PKMeansTests, InitClusters);
        FRIEND_TEST (PKMeansTests, FindClosestCluster);
        FRIEND_TEST (PKMeansTests, AssignDistributions);
        FRIEND_TEST (PKMeansTests, ClearClusterAssignments);
        FRIEND_TEST (PKMeansTests, ComputeClusterMean);
        FRIEND_TEST (PKMeansTests, ComputeNewClusters);
        FRIEND_TEST (PKMeansTests, CalcObjFn);
    public:
        void run (int numClusters, int numThreads, std::string inFilename,
                  std::string assignmentsOutFilename, std::string clustersOutFilename);
};
}

#endif // __PKMEANS_H_INCLUDED_
