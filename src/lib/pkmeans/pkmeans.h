#ifndef __PKMEANS_H_INCLUDED_
#define __PKMEANS_H_INCLUDED_

#include <gtest/gtest.h>
#include <pkmeans/distribution.h>
#include <string>
#include <pthread.h>
#include <cstdint>

namespace pkmeans {
struct ThreadArgs {
    void *_this = nullptr;
    size_t start = 0;
    size_t end = 0;

    ThreadArgs (void *__this, size_t _start, size_t _end) {
        _this = __this;
        start = _start;
        end = _end;
    };
};

class PKMeans {
    private:
        std::vector<Distribution<float>> clusters;
        std::vector<Distribution<float>> newClusters;
        std::vector<Distribution<float>> distributions;
        std::vector<std::vector<size_t>> clusterAssignments;
        std::vector<size_t> clusterMap;
        std::vector<std::vector<std::uint8_t>> lowerBounds;
        std::vector<std::uint8_t> upperBounds;
        std::vector<std::vector<std::uint8_t>> clusterDists;
        std::vector<std::uint8_t> sDists;
        std::vector<bool> upperBoundNeedsUpdate;
        std::vector<pthread_t> threads;
        std::vector<ThreadArgs> threadArgs;
        pthread_attr_t threadAttr;
        bool converged = false;
        size_t denom = 1;

        void readDistributions (std::string inFileName);
        void saveClusters (std::string outFilename);
        void saveAssignments (std::string outFilename);
        void initThreads (int numThreads);
        void startThread (size_t tid, void* (*fn)(void*));
        void runThreads (size_t size, void* (*fn)(void*));
        void joinThreads ();
        void initClusters (int numClusters);
        void initNewClusters ();
        void initLowerBounds ();
        void initUpperBounds ();
        void initAssignments ();
        void initClusterDists ();
        void initSDists ();
        void initUpperBoundNeedsUpdate ();
        void pushClusterDist ();
        void pushSDist ();
        void pushLowerBound ();
        void pushCluster (size_t x);
        void computeClusterDists ();
        void computeSDists ();
        void resetUpperBoundNeedsUpdate ();
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
        std::uint8_t computeDcDist (size_t x, size_t c);
        std::uint8_t cDist (size_t c1, size_t c2);
        float calcObjFn ();
        static void* assignDistributionsThread (void *args);
        static void* computeNewClustersThread (void *args);
        static void* computeLowerBoundsThread (void *args);
        static void* computeUpperBoundsThread (void *args);
        static void* resetUpperBoundNeedsUpdateThread (void *args);

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
        FRIEND_TEST (PKMeansTests, InitUpperBoundNeedsUpdate);
        FRIEND_TEST (PKMeansTests, InitAssignments);
        FRIEND_TEST (PKMeansTests, InitUpperBounds);
        FRIEND_TEST (PKMeansTests, ComputeClusterDists);
        FRIEND_TEST (PKMeansTests, ResetUpperBoundNeedsUpdate);
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
