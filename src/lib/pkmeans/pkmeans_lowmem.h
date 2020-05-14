#ifndef __PKMEANS_LOWMEM_H_INCLUDED_
#define __PKMEANS_LOWMEM_H_INCLUDED_

#include <gtest/gtest.h>
#include <pkmeans/distribution.h>

#include <string>

namespace pkmeans {
class PKMeansLowMem {
 private:
  std::vector<Distribution<float>> clusters;
  std::vector<Distribution<float>> distributions;
  std::vector<std::vector<size_t>> clusterAssignments;
  std::vector<size_t> clusterMap;
  bool converged = false;

  void readDistributions(std::string inFileName);
  void saveClusters(std::string outFilename);
  void saveAssignments(std::string outFilename);
  void initClusters(int numClusters);
  void clearClusterAssignments();
  void assignDistributions();
  void initAssignments();
  void computeNewClusters();
  void computeClusterMean(size_t c);
  float calcObjFn();
  size_t findClosestCluster(size_t x);
  size_t findClosestInitCluster(size_t x);
  size_t getCluster(size_t x);

 public:
  void run(int numClusters, int numThreads, std::string inFilename,
           std::string assignmentsOutFilename, std::string clustersOutFilename);
};
}  // namespace pkmeans

#endif  // __PKMEANS_LOWMEM_H_INCLUDED_