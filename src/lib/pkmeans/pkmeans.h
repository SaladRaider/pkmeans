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
        void computeClusterMean (size_t c);
        bool needsClusterUpdateApprox (size_t x, size_t c);
        bool needsClusterUpdate (size_t x, size_t c);
        size_t findClosestCluster (size_t x);
        size_t findClosestInitCluster (size_t x);
        size_t getCluster (size_t x);
        double computeDcDist (size_t x, size_t c);
        double cDist (size_t c1, size_t c2);
        double calcObjFn ();

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

        FRIEND_TEST (PKMeansTests, InitNewClusters);
        FRIEND_TEST (PKMeansTests, InitLowerBounds);
        FRIEND_TEST (PKMeansTests, InitClusterDists);
        FRIEND_TEST (PKMeansTests, InitSDists);
        FRIEND_TEST (PKMeansTests, InitR);
        FRIEND_TEST (PKMeansTests, InitAssignments);
        FRIEND_TEST (PKMeansTests, InitUpperBounds);
        FRIEND_TEST (PKMeansTests, ComputeClusterDists);
        FRIEND_TEST (PKMeansTests, ResetR);
        FRIEND_TEST (PKMeansTests, AssignNewClusters);
        FRIEND_TEST (PKMeansTests, ComputeLowerBounds);
        FRIEND_TEST (PKMeansTests, ComputeUpperBounds);
        FRIEND_TEST (PKMeansTests, NeedsClusterUpdate);
        FRIEND_TEST (PKMeansTests, FindClosestInitCluster);
        FRIEND_TEST (PKMeansTests, GetCluster);
        FRIEND_TEST (PKMeansTests, ComputeDcDist);
        FRIEND_TEST (PKMeansTests, CDist);
    public:
        void run (int numClusters, int numThreads, std::string inFilename,
                  std::string assignmentsOutFilename, std::string clustersOutFilename);
};
}

#endif // __PKMEANS_H_INCLUDED_
