#include "pkmeans_lowmem.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fstream>
#include <string>
#include <unordered_set>

using namespace pkmeans;

void PKMeansLowMem::run(int numClusters, int numThreads, std::string inFilename,
                        std::string assignmentsOut, std::string clustersOut) {
  printf(
      "Running pkmeans (low memory) with args (k=%d, t=%d, i=%s, a=%s, c=%s)\n",
      numClusters, numThreads, inFilename.c_str(), assignmentsOut.c_str(),
      clustersOut.c_str());

  unsigned int numIterations = 1;
  readDistributions(inFilename);
  initClusters(numClusters);
  assignDistributions();
  printf("Error is %f\n", calcObjFn());
  computeNewClusters();
  while (!converged) {
    assignDistributions();
    printf("Error is %f\n", calcObjFn());
    computeNewClusters();
    numIterations += 1;
  }
  saveAssignments(assignmentsOut);
  saveClusters(clustersOut);

  printf("pkmeans finished running with %u iterations.\n", numIterations);
}

void PKMeansLowMem::readDistributions(std::string inFilename) {
  const auto BUFFER_SIZE = 16 * 1024;
  const auto STR_SIZE = 64;
  int fd = open(inFilename.c_str(), O_RDONLY);
  if (fd == -1) {
    fprintf(stderr, "error opening file %s\n", inFilename.c_str());
    return;
  }

  char buf[BUFFER_SIZE + 1];
  char floatStr[STR_SIZE];
  char *p;
  char *w;
  float bucketVal;
  Distribution<float> newDistribution;

  while (size_t bytes_read = read(fd, buf, BUFFER_SIZE)) {
    if (bytes_read == size_t(-1))
      fprintf(stderr, "read failed on file %s\n", inFilename.c_str());
    if (!bytes_read) break;
    p = buf;
    w = &floatStr[0];
    memset(floatStr, '\0', sizeof(char) * STR_SIZE);
    for (p = buf; p < buf + bytes_read; ++p) {
      if (*p == '\n') {
        bucketVal = atof(floatStr);
        newDistribution.emplace_back(bucketVal);
        distributions.emplace_back(newDistribution);
        clusterMap.emplace_back(size_t(0));
        newDistribution.clear();
        memset(floatStr, '\0', sizeof(char) * STR_SIZE);
        w = &floatStr[0];
      } else if (*p == ' ') {
        bucketVal = atof(floatStr);
        newDistribution.emplace_back(bucketVal);
        memset(floatStr, '\0', sizeof(char) * STR_SIZE);
        w = &floatStr[0];
      } else {
        memcpy(w++, p, sizeof(char) * 1);
      }
    }
  }
  if (newDistribution.size() != 0) {
    distributions.emplace_back(newDistribution);
    clusterMap.emplace_back(size_t(0));
  }
}

void PKMeansLowMem::saveClusters(std::string outFilename) {
  std::ofstream f;
  f.open(outFilename);
  for (size_t i = 0; i < clusters.size(); i++) {
    f << clusters[i] << '\n';
  }
  f.close();
}

void PKMeansLowMem::saveAssignments(std::string outFilename) {
  std::ofstream f;
  f.open(outFilename);
  for (size_t i = 0; i < clusterAssignments.size(); i++) {
    for (size_t j = 0; j < clusterAssignments[i].size(); j++) {
      f << i << ' ' << clusterAssignments[i][j] << '\n';
    }
  }
  f.close();
}

void PKMeansLowMem::initClusters(int numClusters) {
  size_t kMax = size_t(numClusters);
  size_t x = size_t(rand() % int(distributions.size()));
  float p;
  float pSum;
  Distribution<float> weightedP;

  weightedP.fill(0.0, distributions.size());

  // pick random 1st cluster
  clusters.emplace_back(distributions[x]);
  clusterAssignments.emplace_back();

  for (size_t k = 1; k < kMax; k++) {
    initAssignments();

    // calculate weighted probabillities
    for (x = 0; x < distributions.size(); x++)
      weightedP[x] = Distribution<float>::emd(distributions[x],
                                              distributions[getCluster(x)]);
    weightedP *= weightedP;
    weightedP /= weightedP.sum();

    // select new cluster based on weighted probabillites
    p = float(rand()) / float(RAND_MAX);
    pSum = 0.0;
    for (size_t i = 0; i < distributions.size(); i++) {
      pSum += weightedP[i];
      if (pSum >= p) {
        clusters.emplace_back(distributions[i]);
        clusterAssignments.emplace_back();
        break;
      }
    }
  }
}

void PKMeansLowMem::clearClusterAssignments() {
  for (size_t i = 0; i < clusterAssignments.size(); i++) {
    clusterAssignments[i].clear();
  }
}

size_t PKMeansLowMem::findClosestInitCluster(size_t x) {
  size_t cx = getCluster(x);
  if (Distribution<float>::emd(distributions[x], clusters[cx]) <
      Distribution<float>::emd(distributions[x],
                               clusters[clusters.size() - 1])) {
    return cx;
  }
  return clusters.size() - 1;
}

size_t PKMeansLowMem::getCluster(size_t x) { return clusterMap[x]; }

size_t PKMeansLowMem::findClosestCluster(size_t x) {
  size_t closestIdx = 0;
  float dist = Distribution<float>::emd(distributions[x], clusters[0]);
  float newDist;
  for (size_t i = 1; i < clusters.size(); i++) {
    newDist = Distribution<float>::emd(distributions[x], clusters[i]);
    if (newDist < dist) {
      closestIdx = i;
      dist = newDist;
    }
  }
  return closestIdx;
}

void PKMeansLowMem::assignDistributions() {
  converged = true;
  clearClusterAssignments();
  size_t closestClusterIdx;
  for (size_t i = 0; i < distributions.size(); i++) {
    closestClusterIdx = findClosestCluster(i);
    clusterAssignments[closestClusterIdx].emplace_back(i);
    if (closestClusterIdx != clusterMap[i]) {
      converged = false;
      clusterMap[i] = closestClusterIdx;
    }
  }
}

void PKMeansLowMem::initAssignments() {
  for (size_t x = 0; x < distributions.size(); x++) {
    clusterMap[x] = findClosestInitCluster(x);
  }
}

void PKMeansLowMem::computeClusterMean(size_t c) {
  clusters[c].fill(0.0);
  for (size_t i = 0; i < clusterAssignments[c].size(); i++) {
    clusters[c] += distributions[clusterAssignments[c][i]];
  }
  clusters[c] /= float(clusterAssignments[c].size());
}

void PKMeansLowMem::computeNewClusters() {
  for (size_t i = 0; i < clusters.size(); i++) {
    computeClusterMean(i);
  }
}

float PKMeansLowMem::calcObjFn() {
  float sum = 0.0;
  for (size_t i = 0; i < clusterAssignments.size(); i++)
    for (size_t j = 0; j < clusterAssignments[i].size(); j++) {
      sum += Distribution<float>::emd(clusters[i],
                                      distributions[clusterAssignments[i][j]]);
    }
  return sum;
}