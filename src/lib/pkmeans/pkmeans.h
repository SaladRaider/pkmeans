#ifndef __PKMEANS_H_INCLUDED_
#define __PKMEANS_H_INCLUDED_

#include <gtest/gtest.h>
#include <pkmeans/distribution.h>
#include <pthread.h>

#include <limits>
#include <string>
#include <unordered_map>

namespace pkmeans {
struct ThreadArgs {
  void* _this = nullptr;
  size_t start = 0;
  size_t end = 0;

  ThreadArgs(void* __this, size_t _start, size_t _end) {
    _this = __this;
    start = _start;
    end = _end;
  };
};

template <class T>
class PKMeans {
 private:
  std::unordered_map<size_t, size_t> numObservedLocalMin;
  std::vector<Distribution<float>> clusters;
  std::vector<Distribution<float>> newClusters;
  std::vector<Distribution<float>> distributions;
  std::vector<std::vector<size_t>> clusterAssignments;
  std::vector<size_t> clusterMap;
  std::vector<std::vector<T>> lowerBounds;
  std::vector<T> upperBounds;
  std::vector<std::vector<T>> clusterDists;
  std::vector<T> newClusterDists;
  std::vector<T> sDists;
  std::vector<bool> upperBoundNeedsUpdate;
  std::vector<pthread_t> threads;
  std::vector<ThreadArgs> threadArgs;
  pthread_attr_t threadAttr;
  bool converged = false;
  bool quiet;
  size_t denom = 1;
  size_t seed;
  size_t numObservedOnce;
  float maxMissingMass;
  float confidenceProb;
  float bestError = std::numeric_limits<float>::max();

  void readDistributions(const std::string& inFileName);
  void saveClusters(const std::string& outFilename);
  void saveAssignments(const std::string& outFilename);
  void initThreads(int numThreads);
  void startThread(size_t tid, void* (*fn)(void*));
  void runThreads(size_t size, void* (*fn)(void*));
  void joinThreads();
  void initClusters(int numClusters);
  void initNewClusters();
  void initLowerBounds(size_t numClusters);
  void initUpperBounds();
  void initAssignments();
  void initClusterDists();
  void initSDists();
  void initUpperBoundNeedsUpdate();
  void pushClusterDist();
  void pushSDist();
  void pushLowerBound();
  void pushCluster(size_t x);
  void computeClusterDists();
  void computeSDists();
  void resetUpperBoundNeedsUpdate();
  void clearClusterAssignments();
  void assignDistributions();
  void assignNewClusters();
  void computeNewClusters();
  void computeLowerBounds();
  void computeUpperBounds();
  void computeClusterMean(size_t c);
  void markClustersObserved(float objError);
  void reset();
  bool needsClusterUpdateApprox(size_t x, size_t c);
  bool needsClusterUpdate(size_t x, size_t c);
  size_t findClosestCluster(size_t x);
  size_t findClosestInitCluster(size_t x);
  size_t getCluster(size_t x);
  size_t hashClusters();
  float getMissingMass();
  T computeDcDist(size_t x, size_t c);
  T cDist(size_t c1, size_t c2);
  float calcObjFn();
  static void* assignDistributionsThread(void* args);
  static void* computeClusterDistsThread(void* args);
  static void* computeNewClustersThread(void* args);
  static void* computeLowerBoundsThread(void* args);
  static void* computeUpperBoundsThread(void* args);
  static void* resetUpperBoundNeedsUpdateThread(void* args);

  // test friend functions
  friend class PKMeansTests;
  FRIEND_TEST(PKMeansTests, ReadDistributions);
  FRIEND_TEST(PKMeansTests, InitClusters);
  FRIEND_TEST(PKMeansTests, FindClosestCluster);
  FRIEND_TEST(PKMeansTests, AssignDistributions);
  FRIEND_TEST(PKMeansTests, ClearClusterAssignments);
  FRIEND_TEST(PKMeansTests, ComputeClusterMean);
  FRIEND_TEST(PKMeansTests, ComputeNewClusters);
  FRIEND_TEST(PKMeansTests, CalcObjFn);

  FRIEND_TEST(PKMeansTests, InitNewClusters);
  FRIEND_TEST(PKMeansTests, InitLowerBounds);
  FRIEND_TEST(PKMeansTests, InitClusterDists);
  FRIEND_TEST(PKMeansTests, InitSDists);
  FRIEND_TEST(PKMeansTests, InitUpperBoundNeedsUpdate);
  FRIEND_TEST(PKMeansTests, InitAssignments);
  FRIEND_TEST(PKMeansTests, InitUpperBounds);
  FRIEND_TEST(PKMeansTests, ComputeClusterDists);
  FRIEND_TEST(PKMeansTests, ResetUpperBoundNeedsUpdate);
  FRIEND_TEST(PKMeansTests, AssignNewClusters);
  FRIEND_TEST(PKMeansTests, ComputeLowerBounds);
  FRIEND_TEST(PKMeansTests, ComputeUpperBounds);
  FRIEND_TEST(PKMeansTests, NeedsClusterUpdate);
  FRIEND_TEST(PKMeansTests, FindClosestInitCluster);
  FRIEND_TEST(PKMeansTests, GetCluster);
  FRIEND_TEST(PKMeansTests, ComputeDcDist);
  FRIEND_TEST(PKMeansTests, CDist);

 public:
  void run(int numClusters, int numThreads, float confidenceProb,
           float maxMissingMass, size_t _seed, bool useTimeSeed,
           const std::string& inFilename,
           const std::string& assignmentsOutFilename,
           const std::string& clustersOutFilename, bool euclidean, bool quiet);
  void runOnce(int numClusters, const std::string& assignmentsOutFilename,
               const std::string& clustersOutFilename);

  template <typename U>
  static T emd(const Distribution<U>& d1, const Distribution<U>& d2,
               size_t denom) {
    U sum = Distribution<U>::emd(d1, d2);
    constexpr T maxVal = sizeof(T) - 1;
    constexpr T halfVal = sizeof(T) / 2 - 1;
    if (sizeof(T) < sizeof(U) && maxVal > 0)
      return (sum * maxVal) / denom;
    else if (sizeof(T) < sizeof(U))
      return (sum * halfVal) / denom;
    else
      return T(sum);
  };

  static T calcDist(const Distribution<T>& d1, const Distribution<T>& d2) {
    return Distribution<T>::emd(d1, d2);
  };
};
}  // namespace pkmeans

#endif  // __PKMEANS_H_INCLUDED_
