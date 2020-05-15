#include "pkmeans.h"

#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <tgmath.h>
#include <omp.h>

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
                     const std::string &clustersOut, bool euclidean,
                     bool _quiet) {
  if (useTimeSeed)
    seed = time(NULL);
  else
    seed = _seed;
  confidenceProb = _confidenceProb;
  maxMissingMass = _maxMissingMass;
  Distribution<float>::euclidean = euclidean;
  quiet = _quiet;
  printf(
      "Running pkmeans with args (k=%d, t=%d, p=%f, m=%f, s=%ld, i=%s, a=%s, "
      "c=%s, e=%s, q=%s)\n",
      numClusters, numThreads, _confidenceProb, _maxMissingMass, seed,
      inFilename.c_str(), assignmentsOut.c_str(), clustersOut.c_str(),
      euclidean ? "true" : "false", quiet ? "true" : "false");

  readDistributions(inFilename);
  initThreads(numThreads);
  initLowerBounds(numClusters);

  unsigned int numIterations = 0;
  numObservedLocalMin.clear();
  numObservedOnce = 0;
  bestError = std::numeric_limits<float>::max();
  do {
    reset();
    runOnce(numClusters, assignmentsOut, clustersOut);
    numIterations += 1;
  } while (maxMissingMass <= getMissingMass());
  pthread_attr_destroy(&threadAttr);

  if (quiet) {
    std::cout << '\n';
  }
  printf("best error: %f\n", bestError);
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
    if (quiet)
      printf("\nFound new best error: %f; seed: %ld; saving assignments...\n",
             bestError, seed);
    else
      printf("Found new best error: %f; seed: %ld; saving assignments...\n",
             bestError, seed);
  }
  markClustersObserved(objError);
  if (!quiet) {
    printf("best error: %f\n", bestError);
    printf("finished one run with seed %ld and %u iterations.\n", seed,
           numIterations);
  }
  seed += 39916801;
}

template <class T>
void PKMeans<T>::markClustersObserved(float objError) {
  size_t h = size_t(objError);
  numObservedLocalMin[h] += 1;
  if (numObservedLocalMin[h] == 1)
    numObservedOnce += 1;
  else if (numObservedLocalMin[h] == 2)
    numObservedOnce -= 1;
  if (!quiet)
    printf("numObservedLocalMin[%zu] = %zu\n", hashClusters(),
           numObservedLocalMin[hashClusters()]);
}

template <class T>
size_t PKMeans<T>::hashClusters() {
  std::vector<size_t> clusterHashes;
  for (auto c = 0; c < clusters.size(); c++) {
    std::sort(clusterAssignments[c].begin(), clusterAssignments[c].end());
    clusterHashes.emplace_back(boost::hash_range(clusterAssignments[c].begin(),
                                                 clusterAssignments[c].end()));
  }
  std::sort(clusterHashes.begin(), clusterHashes.end());
  return boost::hash_range(clusterHashes.begin(), clusterHashes.end());
}

template <class T>
float PKMeans<T>::getMissingMass() {
  auto G = float(numObservedOnce) / float(numObservedLocalMin.size());
  constexpr auto A =
      2.f * 1.41421356237f + 1.73205080757f;  // 2*sqrt(2) + sqrt(3)
  auto B = A * sqrt(log(3.f / confidenceProb) / numObservedLocalMin.size());
  auto missingMass = G + B;
  if (quiet) {
    printf("\rG: %f, B: %f, missingMass: %f, maxMissingMass: %f n: %ld", G, B,
           missingMass, maxMissingMass, numObservedLocalMin.size());
    std::cout << std::flush;
  } else
    printf("G: %f, B: %f, missingMass: %f, maxMissingMass: %f n: %ld\n", G, B,
           missingMass, maxMissingMass, numObservedLocalMin.size());
  return missingMass;
}

template <class T>
void PKMeans<T>::reset() {
  clusters.clear();
  newClusters.clear();
  clusterAssignments.clear();
  for (size_t x = 0; x < lowerBounds.size(); x++)
    for (size_t c = 0; c < lowerBounds[x].size(); c++) lowerBounds[x][c] = 0;
  upperBounds.clear();
  clusterDists.clear();
  sDists.clear();
  upperBoundNeedsUpdate.clear();
  converged = false;
}

template <class T>
void PKMeans<T>::initThreads(int numThreads) {
  threads.clear();
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
  distributions.clear();
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
      } else if (*p == ' ' || *p == '\t') {
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
  #pragma omp parallel for
  for (size_t x = 0; x < distributions.size(); x++) {
    findClosestCluster(x);
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
  #pragma omp parallel for
  for (size_t c = 0; c < clusters.size(); c++) {
    computeClusterMean(c);
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
    sum += PKMeans<float>::calcDist(distributions[x], clusters[getCluster(x)]);
  }
  return sum;
}

template <class T>
void PKMeans<T>::initLowerBounds(size_t numClusters) {
  lowerBounds.clear();
  for (size_t x = 0; x < distributions.size(); x++) {
    lowerBounds.emplace_back();
    for (size_t c = 0; c < numClusters; c++) {
      lowerBounds[x].emplace_back(0);
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
    lowerBounds[x][clusters.size() - 1] = 0;
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
  #pragma omp parallel for
  for (size_t x = 0; x < distributions.size(); x++) {
    upperBoundNeedsUpdate[x] = true;
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
  #pragma omp parallel for
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
  T dist;
  #pragma omp parallel for private(dist)
  for (size_t c = 0; c < clusters.size(); c++) {
    dist = PKMeans<T>::emd<float>(clusters[c], newClusters[c], denom);
    for (size_t x = 0; x < distributions.size(); x++)
      lowerBounds[x][c] = fmax(lowerBounds[x][c] - dist, 0);
  }
}

template <class T>
void *PKMeans<T>::computeLowerBoundsThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  T dist;
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
  T dist;
  #pragma omp parallel for private(dist)
  for (size_t c = 0; c < clusters.size(); c++) {
    dist = PKMeans<T>::emd<float>(clusters[c], newClusters[c], denom);
    for (size_t i = 0; i < clusterAssignments[c].size(); i++) {
      upperBounds[clusterAssignments[c][i]] += dist;
    }
  }
}

template <class T>
void *PKMeans<T>::computeUpperBoundsThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  T dist;
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
  #pragma omp parallel for
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
  if (upperBoundNeedsUpdate[x]) {
    upperBounds[x] = computeDcDist(x, cx);
    upperBoundNeedsUpdate[x] = false;
  }
  return (upperBounds[x] > lowerBounds[x][c] || upperBounds[x] > 0.5 * cDist(cx, c)) &&
         computeDcDist(x, c) < upperBounds[x];
}

namespace pkmeans {
template class PKMeans<float>;
template class PKMeans<std::uint8_t>;
}  // namespace pkmeans
