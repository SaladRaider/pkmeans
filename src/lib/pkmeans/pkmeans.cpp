#include "pkmeans.h"

#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <tgmath.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <string>

using namespace pkmeans;

template <class T>
void PKMeans<T>::run(int numClusters, int numThreads, float _confidenceProb,
                     float _maxMissingMass, size_t _seed, bool useTimeSeed,
                     const std::string &inFilename,
                     const std::string &assignmentsOut,
                     const std::string &clustersOut, bool _quiet) {
  if (useTimeSeed)
    seed = time(NULL);
  else
    seed = _seed;
  confidenceProb = _confidenceProb;
  maxMissingMass = _maxMissingMass;
  quiet = _quiet;
  printf(
      "Running pkmeans with args (k=%d, t=%d, p=%f, m=%f, s=%ld, i=%s, a=%s, "
      "c=%s)\n",
      numClusters, numThreads, _confidenceProb, _maxMissingMass, seed,
      inFilename.c_str(), assignmentsOut.c_str(), clustersOut.c_str());

  readDistributions(inFilename);
  initThreads(numThreads);

  unsigned int numIterations = 0;
  bestError = std::numeric_limits<float>::max();
  do {
    reset();
    runOnce(numClusters, assignmentsOut, clustersOut);
    numIterations += 1;
  } while (maxMissingMass <= getMissingMass());
  pthread_attr_destroy(&threadAttr);

  printf("pkmeans finished running with %u restarts.\n", numIterations);
}

template <class T>
void PKMeans<T>::runOnce(int numClusters, const std::string &assignmentsOut,
                         const std::string &clustersOut) {
  // set rand seed
  srand(seed);
  if (!quiet) printf("Starting kmeans\n");

  unsigned int numIterations = 1;
  if (!quiet) printf("done reading file.\n");
  initClusters(numClusters);
  if (!quiet) printf("done intializing centroids.\n");
  initNewClusters();
  initUpperBoundNeedsUpdate();
  initUpperBounds();
  if (!quiet) printf("done intializing.\n");
  computeNewClusters();
  computeLowerBounds();
  computeUpperBounds();
  resetUpperBoundNeedsUpdate();
  assignNewClusters();
  auto objError = calcObjFn();
  if (!quiet) printf("Error is %f\n", objError);
  while (!converged) {
    computeClusterDists();
    assignDistributions();
    computeNewClusters();
    computeLowerBounds();
    computeUpperBounds();
    resetUpperBoundNeedsUpdate();
    assignNewClusters();
    objError = calcObjFn();
    if (!quiet) printf("Error is %f\n", objError);
    numIterations += 1;
  }
  if (objError < bestError) {
    bestError = objError;
    saveAssignments(assignmentsOut);
    saveClusters(clustersOut);
  }
  markClustersObserved();
  if (!quiet) printf("best error: %f\n", bestError);
  if (!quiet)
    printf("finished one run with seed %ld and %u iterations.\n", seed,
           numIterations);
  seed += 1;
}

template <class T>
void PKMeans<T>::markClustersObserved() {
  numObservedLocalMin[hashClusters()] += 1;
  if (!quiet)
    printf("numObservedLocalMin[%zu] = %zu\n", hashClusters(),
           numObservedLocalMin[hashClusters()]);
}

template <class T>
size_t PKMeans<T>::hashClusters() {
  std::vector<size_t> clusterHashes;
  for (auto c = 0; c < clusters.size(); c++) {
    clusterHashes.emplace_back(clusters[c].hash());
  }
  std::sort(clusterHashes.begin(), clusterHashes.end());
  return boost::hash_range(clusterHashes.begin(), clusterHashes.end());
}

template <class T>
float PKMeans<T>::getMissingMass() {
  auto G = 0.f;
  constexpr auto A =
      2.f * 1.41421356237f + 1.73205080757f;  // 2*sqrt(2) + sqrt(3)
  for (auto e : numObservedLocalMin) G += e.second == 1;
  G /= numObservedLocalMin.size();
  auto B = A * sqrt(log(3.f / confidenceProb) / numObservedLocalMin.size());
  auto missingMass = G / numObservedLocalMin.size() + B;
  printf("G: %f, B: %f, missingMass: %f, maxMissingMass: %f n: %ld\n", G, B,
         missingMass, maxMissingMass, numObservedLocalMin.size());
  return missingMass;
}

template <class T>
void PKMeans<T>::reset() {
  clusters.clear();
  newClusters.clear();
  clusterAssignments.clear();
  lowerBounds.clear();
  upperBounds.clear();
  clusterDists.clear();
  sDists.clear();
  upperBoundNeedsUpdate.clear();
  converged = false;
}

template <class T>
void PKMeans<T>::initThreads(int numThreads) {
  for (size_t i = 0; i < numThreads; i++) {
    threads.emplace_back();
    threadArgs.emplace_back((void *)this, 0, 0);
  }
  pthread_attr_init(&threadAttr);
  pthread_attr_setdetachstate(&threadAttr, PTHREAD_CREATE_JOINABLE);
}

template <class T>
void PKMeans<T>::startThread(size_t tid, void *(*fn)(void *)) {
  int rc = pthread_create(&threads[tid], &threadAttr, fn,
                          (void *)&(threadArgs[tid]));
  if (rc) {
    std::cout << "Error: unable to create thread," << rc << '\n';
    exit(-1);
  }
}

template <class T>
void PKMeans<T>::runThreads(size_t size, void *(*fn)(void *)) {
  size_t itemsPerThread = size / threads.size();
  size_t i = 0;
  size_t tid = 0;
  for (tid = 0; tid < threads.size() - 1; tid++) {
    threadArgs[tid].start = i;
    i += itemsPerThread;
    threadArgs[tid].end = i;
    startThread(tid, fn);
  }
  threadArgs[tid].start = i;
  threadArgs[tid].end = size;
  startThread(tid, fn);
  joinThreads();
}

template <class T>
void PKMeans<T>::joinThreads() {
  int rc;
  void *status;
  for (size_t tid = 0; tid < threads.size(); tid++) {
    rc = pthread_join(threads[tid], &status);
    if (rc) {
      std::cout << "Error: unable to join," << rc << '\n';
      exit(-1);
    }
  }
}

template <class T>
void PKMeans<T>::readDistributions(const std::string &inFilename) {
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
  denom = (distributions[0].size() - 1) * distributions[0].sum();
}

template <class T>
void PKMeans<T>::saveClusters(const std::string &outFilename) {
  std::ofstream f;
  f.open(outFilename);
  for (size_t c = 0; c < clusters.size(); c++) {
    f << clusters[c] << '\n';
  }
  f.close();
}

template <class T>
void PKMeans<T>::saveAssignments(const std::string &outFilename) {
  std::ofstream f;
  f.open(outFilename);
  for (size_t c = 0; c < clusterAssignments.size(); c++) {
    for (size_t j = 0; j < clusterAssignments[c].size(); j++) {
      f << c << ' ' << clusterAssignments[c][j] << '\n';
    }
  }
  f.close();
}

template <class T>
void PKMeans<T>::initClusters(int numClusters) {
  size_t kMax = size_t(numClusters);
  size_t x = size_t(rand() % int(distributions.size()));
  float p;
  float pSum;
  Distribution<float> weightedP;

  weightedP.fill(0.0, distributions.size());
  initLowerBounds();

  // pick random 1st cluster
  pushCluster(x);

  for (size_t k = 1; k < kMax; k++) {
    // calculate weighted probabillities
    for (x = 0; x < distributions.size(); x++)
      weightedP[x] = lowerBounds[x][getCluster(x)];
    weightedP *= weightedP;
    weightedP /= weightedP.sum();

    // select new cluster based on weighted probabillites
    p = float(rand()) / float(RAND_MAX);
    pSum = 0.0;
    for (x = 0; x < distributions.size(); x++) {
      pSum += weightedP[x];
      if (pSum >= p) {
        pushCluster(x);
        break;
      }
    }
  }

  for (size_t x = 0; x < distributions.size(); x++) {
    clusterAssignments[getCluster(x)].emplace_back(x);
  }
}

template <class T>
void PKMeans<T>::initNewClusters() {
  for (size_t c = 0; c < clusters.size(); c++)
    newClusters.emplace_back(clusters[c]);
}

template <class T>
void PKMeans<T>::clearClusterAssignments() {
  for (size_t c = 0; c < clusterAssignments.size(); c++) {
    clusterAssignments[c].clear();
  }
}

template <class T>
size_t PKMeans<T>::findClosestCluster(size_t x) {
  if (upperBounds[x] <= sDists[getCluster(x)]) return getCluster(x);
  for (size_t c = 0; c < clusters.size(); c++) {
    if (needsClusterUpdateApprox(x, c) && needsClusterUpdate(x, c)) {
      converged = false;
      clusterMap[x] = c;
    }
  }
  return getCluster(x);
}

template <class T>
size_t PKMeans<T>::findClosestInitCluster(size_t x) {
  // if 1/2 d(c,c') >= d(x,c), then
  //     d(x,c') >= d(x,c)
  if (clusters.size() == 1) {
    computeDcDist(x, 0);
    return 0;
  }
  size_t cx = getCluster(x);
  if (0.5 * cDist(cx, clusters.size() - 1) >= lowerBounds[x][cx] ||
      lowerBounds[x][cx] < computeDcDist(x, clusters.size() - 1)) {
    return cx;
  }
  return clusters.size() - 1;
}

template <class T>
size_t PKMeans<T>::getCluster(size_t x) {
  return clusterMap[x];
}

template <class T>
void PKMeans<T>::assignDistributions() {
  converged = true;
  clearClusterAssignments();
  if (threads.size() > 1) {
    runThreads(distributions.size(), PKMeans::assignDistributionsThread);
  } else {
    for (size_t x = 0; x < distributions.size(); x++) {
      findClosestCluster(x);
    }
  }
  for (size_t x = 0; x < distributions.size(); x++) {
    clusterAssignments[getCluster(x)].emplace_back(x);
  }
}

template <class T>
void *PKMeans<T>::assignDistributionsThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  for (size_t x = threadArgs->start; x < threadArgs->end; x++) {
    pkmeans->findClosestCluster(x);
  }
  pthread_exit(NULL);
}

template <class T>
void PKMeans<T>::computeClusterMean(size_t c) {
  newClusters[c].fill(0.0);
  for (size_t i = 0; i < clusterAssignments[c].size(); i++) {
    newClusters[c] += distributions[clusterAssignments[c][i]];
  }
  newClusters[c] /= float(clusterAssignments[c].size());
}

template <class T>
void PKMeans<T>::computeNewClusters() {
  if (threads.size() > 1) {
    runThreads(clusters.size(), PKMeans::computeNewClustersThread);
  } else {
    for (size_t c = 0; c < clusters.size(); c++) {
      computeClusterMean(c);
    }
  }
}

template <class T>
void *PKMeans<T>::computeNewClustersThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  for (size_t c = threadArgs->start; c < threadArgs->end; c++) {
    pkmeans->computeClusterMean(c);
  }
  pthread_exit(NULL);
}

template <class T>
float PKMeans<T>::calcObjFn() {
  float sum = 0.0;
  for (size_t x = 0; x < distributions.size(); x++) {
    sum += Distribution<float>::emd(distributions[x], clusters[getCluster(x)]);
  }
  return sum;
}

template <class T>
void PKMeans<T>::initLowerBounds() {
  lowerBounds.clear();
  for (size_t x = 0; x < distributions.size(); x++) {
    lowerBounds.emplace_back();
    for (size_t c = 0; c < clusters.size(); c++) {
      lowerBounds[x].emplace_back(0.0);
    }
  }
}

template <class T>
void PKMeans<T>::initUpperBounds() {
  upperBounds.clear();
  for (size_t x = 0; x < distributions.size(); x++) {
    upperBounds.emplace_back(computeDcDist(x, getCluster(x)));
  }
}

template <class T>
void PKMeans<T>::initAssignments() {
  for (size_t x = 0; x < distributions.size(); x++) {
    clusterMap[x] = findClosestInitCluster(x);
  }
}

template <class T>
void PKMeans<T>::initClusterDists() {
  clusterDists.clear();
  for (size_t c1 = 0; c1 < clusters.size(); c1++) {
    clusterDists.emplace_back();
    for (size_t c2 = 0; c2 < clusters.size(); c2++)
      clusterDists[c1].emplace_back(
          PKMeans<T>::emd<float>(clusters[c1], clusters[c2], denom));
  }
}

template <class T>
void PKMeans<T>::initSDists() {
  sDists.clear();
  for (size_t c = 0; c < clusters.size(); c++) {
    sDists.emplace_back();
  }
}

template <class T>
void PKMeans<T>::initUpperBoundNeedsUpdate() {
  upperBoundNeedsUpdate.clear();
  for (size_t x = 0; x < distributions.size(); x++) {
    upperBoundNeedsUpdate.emplace_back(true);
  }
}

template <class T>
void PKMeans<T>::pushClusterDist() {
  size_t cNew = clusters.size() - 1;
  for (size_t c = 0; c < cNew; c++) {
    clusterDists[c].emplace_back(
        PKMeans<T>::emd<float>(clusters[c], clusters[cNew], denom));
  }
  clusterDists.emplace_back();
  for (size_t c = 0; c < clusters.size(); c++) {
    clusterDists[cNew].emplace_back(
        PKMeans<T>::emd<float>(clusters[cNew], clusters[c], denom));
  }
}

template <class T>
void PKMeans<T>::pushSDist() {
  sDists.emplace_back();
}

template <class T>
void PKMeans<T>::pushLowerBound() {
  for (size_t x = 0; x < distributions.size(); x++) {
    lowerBounds[x].emplace_back(0.0);
  }
}

template <class T>
void PKMeans<T>::pushCluster(size_t x) {
  clusters.emplace_back(distributions[x]);
  clusterAssignments.emplace_back();
  pushClusterDist();
  pushSDist();
  pushLowerBound();
  initAssignments();
}

template <class T>
void PKMeans<T>::resetUpperBoundNeedsUpdate() {
  if (threads.size() > 1) {
    runThreads(distributions.size(), PKMeans::resetUpperBoundNeedsUpdateThread);
  } else {
    for (size_t x = 0; x < distributions.size(); x++) {
      upperBoundNeedsUpdate[x] = true;
    }
  }
}

template <class T>
void *PKMeans<T>::resetUpperBoundNeedsUpdateThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  for (size_t x = threadArgs->start; x < threadArgs->end; x++) {
    pkmeans->upperBoundNeedsUpdate[x] = true;
  }
  pthread_exit(NULL);
}

template <class T>
void PKMeans<T>::computeClusterDists() {
  if (threads.size() > 1) {
    runThreads(clusters.size(), PKMeans::computeClusterDistsThread);
  } else {
    for (size_t c1 = 0; c1 < clusters.size(); c1++) {
      sDists[c1] = 255;
      for (size_t c2 = 0; c2 < clusters.size(); c2++) {
        clusterDists[c1][c2] =
            PKMeans<T>::emd<float>(clusters[c1], clusters[c2], denom);
        if (c1 != c2 && 0.5 * clusterDists[c1][c2] < sDists[c1])
          sDists[c1] = 0.5 * clusterDists[c1][c2];
      }
    }
  }
}

template <class T>
void *PKMeans<T>::computeClusterDistsThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  for (size_t c1 = threadArgs->start; c1 < threadArgs->end; c1++) {
    pkmeans->sDists[c1] = 255;
    for (size_t c2 = 0; c2 < pkmeans->clusters.size(); c2++) {
      pkmeans->clusterDists[c1][c2] = PKMeans<T>::emd<float>(
          pkmeans->clusters[c1], pkmeans->clusters[c2], pkmeans->denom);
      if (c1 != c2 && 0.5 * pkmeans->clusterDists[c1][c2] < pkmeans->sDists[c1])
        pkmeans->sDists[c1] = 0.5 * pkmeans->clusterDists[c1][c2];
    }
  }
  pthread_exit(NULL);
}

template <class T>
void PKMeans<T>::computeLowerBounds() {
  if (threads.size() > 1) {
    runThreads(clusters.size(), PKMeans::computeLowerBoundsThread);
  } else {
    std::uint8_t dist;
    for (size_t c = 0; c < clusters.size(); c++) {
      dist = PKMeans<T>::emd<float>(clusters[c], newClusters[c], denom);
      for (size_t x = 0; x < distributions.size(); x++)
        lowerBounds[x][c] = fmax(lowerBounds[x][c] - dist, 0);
    }
  }
}

template <class T>
void *PKMeans<T>::computeLowerBoundsThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  std::uint8_t dist;
  for (size_t c = threadArgs->start; c < threadArgs->end; c++) {
    dist = PKMeans<T>::emd<float>(pkmeans->clusters[c], pkmeans->newClusters[c],
                                  pkmeans->denom);
    for (size_t x = 0; x < pkmeans->distributions.size(); x++) {
      pkmeans->lowerBounds[x][c] = fmax(pkmeans->lowerBounds[x][c] - dist, 0);
    }
  }
  pthread_exit(NULL);
}

template <class T>
void PKMeans<T>::computeUpperBounds() {
  if (threads.size() > 1) {
    runThreads(clusters.size(), PKMeans::computeUpperBoundsThread);
  } else {
    std::uint8_t dist;
    for (size_t c = 0; c < clusters.size(); c++) {
      dist = PKMeans<T>::emd<float>(clusters[c], newClusters[c], denom);
      for (size_t i = 0; i < clusterAssignments[c].size(); i++) {
        upperBounds[clusterAssignments[c][i]] += dist;
      }
    }
  }
}

template <class T>
void *PKMeans<T>::computeUpperBoundsThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  std::uint8_t dist;
  for (size_t c = threadArgs->start; c < threadArgs->end; c++) {
    dist = PKMeans<T>::emd<float>(pkmeans->clusters[c], pkmeans->newClusters[c],
                                  pkmeans->denom);
    for (size_t i = 0; i < pkmeans->clusterAssignments[c].size(); i++) {
      pkmeans->upperBounds[pkmeans->clusterAssignments[c][i]] += dist;
    }
  }
  pthread_exit(NULL);
}

template <class T>
void PKMeans<T>::assignNewClusters() {
  for (size_t c = 0; c < clusters.size(); c++) clusters[c] = newClusters[c];
}

template <class T>
T PKMeans<T>::computeDcDist(size_t x, size_t c) {
  lowerBounds[x][c] =
      PKMeans<T>::emd<float>(distributions[x], clusters[c], denom);
  return lowerBounds[x][c];
}

template <class T>
T PKMeans<T>::cDist(size_t c1, size_t c2) {
  return clusterDists[c1][c2];
}

template <class T>
bool PKMeans<T>::needsClusterUpdateApprox(size_t x, size_t c) {
  return c != getCluster(x) && upperBounds[x] > lowerBounds[x][c] &&
         upperBounds[x] > 0.5 * cDist(getCluster(x), c);
}

template <class T>
bool PKMeans<T>::needsClusterUpdate(size_t x, size_t c) {
  size_t cx = getCluster(x);
  size_t dcDist;
  if (upperBoundNeedsUpdate[x]) {
    dcDist = computeDcDist(x, cx);
    upperBounds[x] = dcDist;
    upperBoundNeedsUpdate[x] = false;
  } else {
    dcDist = upperBounds[x];
  }
  return (dcDist > lowerBounds[x][c] || dcDist > 0.5 * cDist(cx, c)) &&
         computeDcDist(x, c) < dcDist;
}

namespace pkmeans {
template class PKMeans<float>;
template class PKMeans<std::uint8_t>;
}  // namespace pkmeans
