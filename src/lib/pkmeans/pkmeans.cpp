#include "pkmeans.h"

#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <tgmath.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <string>

#include "progress_bar.h"

using namespace pkmeans;

template <class T>
void PKMeans<T>::run(int _numClusters, int numThreads, float _confidenceProb,
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
  numClusters = _numClusters;
  printf(
      "Running pkmeans with args (k=%ld, t=%d, p=%f, m=%f, l=%s, s=%ld, i=%s, "
      "a=%s, c=%s, e=%s, q=%s)\n",
      numClusters, numThreads, _confidenceProb, _maxMissingMass,
      sizeof(T) == 1 ? "true" : "false", seed, inFilename.c_str(),
      assignmentsOut.c_str(), clustersOut.c_str(), euclidean ? "true" : "false",
      quiet ? "true" : "false");

  readDistributions(inFilename);
  if (!quiet) printf("done reading file.\n");
  printf("initializing threads...\n");
  initThreads(numThreads);
  if (!quiet) printf("done initializing threads.\n");
  printf("initializing lower bounds...\n");
  initLowerBounds();
  if (!quiet) printf("done initializing lower bounds.\n");

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
  printf("pkmeans finished running with %u restarts.\n", numIterations - 1);
}

template <class T>
void PKMeans<T>::runOnce(int numClusters, const std::string &assignmentsOut,
                         const std::string &clustersOut) {
  // set rand seed
  srand(seed);
  if (!quiet) printf("Starting kmeans\n");

  unsigned int numIterations = 1;
  if (!quiet) printf("\033[1minitializing centroids\033[0m\n");
  initClusters();
  if (!quiet) printf("done intializing centroids.\n");
  if (!quiet) printf("initializing run variables...\n");
  computeClusterDists();
  initNewClusters();
  initUpperBounds();
  if (!quiet) printf("done intializing.\n");
  computeNewClusters();
  computeUpperBounds();
  computeLowerBounds();
  assignNewClusters();
  auto objError = calcObjFn();
  if (!quiet) printf("Error is %f\n", objError);
  while (!converged) {
    if (!quiet) printf("computing cluster dists...\n");
    computeClusterDists();
    if (!quiet) printf("assigning...\n");
    assignDistributions();
    if (!quiet) printf("computing new clusters...\n");
    computeNewClusters();
    if (!quiet) printf("computing upper bounds...\n");
    computeUpperBounds();
    if (!quiet) printf("computing lower bounds...\n");
    computeLowerBounds();
    if (!quiet) printf("assigning new clusters...\n");
    assignNewClusters();
    numIterations += 1;
    if (!quiet) printf("done with itteration %u\n", numIterations);
  }
  if (!quiet) printf("calculating obj fn...\n");
  objError = calcObjFn();
  if (!quiet) printf("Error is %f\n", objError);
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
  markClustersObserved();
  if (!quiet) {
    printf("best error: %f\n", bestError);
    printf("finished one run with seed %ld and %u iterations.\n", seed,
           numIterations);
  }
  seed += 39916801;
}

template <class T>
void PKMeans<T>::markClustersObserved() {
  size_t h = hashClusters();
  numObservedLocalMin[h] += 1;
  if (numObservedLocalMin[h] == 1)
    numObservedOnce += 1;
  else if (numObservedLocalMin[h] == 2)
    numObservedOnce -= 1;
  if (!quiet)
    printf("numObservedLocalMin[%zu] = %zu\n", hashClusters(),
           numObservedLocalMin[h]);
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
  if (!quiet) printf("resetting run variables...\n");
  clusters.clear();
  newClusters.clear();
  clusterAssignments.clear();
  newClusterDists.clear();
  for (size_t x = 0; x < distributions.size(); x++)
    for (size_t c = 0; c < numClusters; c++) getLowerBounds(x, c) = 0;
  upperBounds.clear();
  clusterDists.clear();
  sDists.clear();
  converged = false;
  if (!quiet) printf("done resetting run variables\n");
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
void PKMeans<T>::runThreads(size_t size,
                            const std::function<void *(void *)> &fn) {
  size_t itemsPerThread = size / threads.size();
  size_t i = 0;
  size_t tid = 0;
  for (tid = 0; tid < threads.size() - 1; tid++) {
    threadArgs[tid].fn = &fn;
    threadArgs[tid].start = i;
    i += itemsPerThread;
    threadArgs[tid].end = i;
    startThread(tid, threadFnWrapper);
  }
  threadArgs[tid].fn = &fn;
  threadArgs[tid].start = i;
  threadArgs[tid].end = size;
  startThread(tid, threadFnWrapper);
  joinThreads();
}

template <class T>
void *PKMeans<T>::threadFnWrapper(void *args) {
  return (*((ThreadArgs *)args)->fn)(args);
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
  constexpr auto BUFFER_SIZE = 16 * 1024;
  constexpr auto STR_SIZE = 128;
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

  struct stat filestatus;
  stat(inFilename.c_str(), &filestatus);
  auto filesize = filestatus.st_size;
  auto numBytesRead = size_t{0};
  auto prevProgress = size_t{0};
  auto progressBar = ProgressBar<50>();
  printf("\033[1mreading %s; filesize: %lld\033[0m\n", inFilename.c_str(),
         filesize);

  w = &floatStr[0];
  while (size_t bytes_read = read(fd, buf, BUFFER_SIZE)) {
    if (bytes_read == size_t(-1))
      fprintf(stderr, "read failed on file %s\n", inFilename.c_str());
    if (!bytes_read) break;
    p = buf;
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
    // write progress bar
    numBytesRead += BUFFER_SIZE;
    progressBar.show(float(numBytesRead) / filesize);
  }
  std::cout << '\n';
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
void PKMeans<T>::initClusters() {
  size_t kMax = size_t(numClusters);
  size_t x = size_t(rand() % int(distributions.size()));
  size_t newClusterIdx;
  double p;
  double weightedSum;
  std::vector<double> weightedP(distributions.size(), 0.0);
  std::vector<double> weightedSums(threads.size(), 0.0);
  ProgressBar<50> progressBar;
  if (!quiet) progressBar.show(0.f);

  // pick random 1st cluster
  pushCluster(x);
  if (!quiet) progressBar.show(1.f / kMax);

  std::function<void *(void *)> calcWeighted =
      [&weightedP, &weightedSums](void *args) -> void * {
    ThreadArgs *threadArgs = (ThreadArgs *)args;
    PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
    auto i = size_t(threadArgs->start / pkmeans->distributions.size());
    for (size_t x = threadArgs->start; x < threadArgs->end; x++) {
      weightedP[x] = pkmeans->getLowerBounds(x, pkmeans->getCluster(x));
      weightedP[x] *= weightedP[x];
      weightedSums[i] += weightedP[x];
    }
    pthread_exit(NULL);
  };
  for (size_t k = 1; k < kMax; k++) {
    // calculate weighted probabillities
    weightedSum = 0;
    if (threads.size() > 1) {
      runThreads(distributions.size(), calcWeighted);
      p = double(rand()) / double(RAND_MAX);
      for (auto i = 0; i < weightedSums.size(); i++) {
        weightedSum += weightedSums[i];
      }
      // select new cluster based on weighted probabillites
      p = (double(rand()) / double(RAND_MAX)) * weightedSum;
      for (auto i = 0; i < weightedSums.size(); i++) {
        p -= weightedSums[i];
        if (p <= 0) {
          p += weightedSums[i];
          for (x = i * (distributions.size() / threads.size());
               x < distributions.size(); x++) {
            p -= weightedP[x];
            if (p <= 0) {
              pushCluster(x);
              break;
            }
          }
          break;
        }
      }
    } else {
      for (size_t x = 0; x < distributions.size(); x++) {
        weightedP[x] = getLowerBounds(x, getCluster(x));
        weightedP[x] *= weightedP[x];
        weightedSum += weightedP[x];
      }
      // select new cluster based on weighted probabillites
      p = (double(rand()) / double(RAND_MAX)) * weightedSum;
      for (x = 0; x < distributions.size(); x++) {
        p -= weightedP[x];
        if (p <= 0) {
          pushCluster(x);
          break;
        }
      }
    }
    if (!quiet) progressBar.show((k + 1.f) / kMax);
  }
  for (size_t x = 0; x < distributions.size(); x++) {
    clusterAssignments[getCluster(x)].emplace_back(x);
  }
  if (!quiet) printf("\n");
}

template <class T>
inline void PKMeans<T>::initNewClusters() {
  for (size_t c = 0; c < clusters.size(); c++)
    newClusters.emplace_back(clusters[c]);
}

template <class T>
inline void PKMeans<T>::clearClusterAssignments() {
  for (size_t c = 0; c < clusterAssignments.size(); c++) {
    clusterAssignments[c].clear();
  }
}

template <class T>
inline size_t PKMeans<T>::findClosestCluster(size_t x) {
  if (upperBounds[x] <= sDists[getCluster(x)]) return getCluster(x);
  upperBounds[x] = computeDcDist(x, getCluster(x));
  for (size_t c = 0; c < clusters.size(); c++) {
    if (needsClusterUpdateApprox(x, c) && needsClusterUpdate(x, c)) {
      converged = false;
      clusterMap[x] = c;
      upperBounds[x] = computeDcDist(x, c);
    }
  }
  return getCluster(x);
}

template <class T>
inline size_t PKMeans<T>::findClosestInitCluster(size_t x) {
  // if 1/2 d(c,c') >= d(x,c), then
  //     d(x,c') >= d(x,c)
  if (cDist(getCluster(x), clusters.size() - 1) >=
          getLowerBounds(x, getCluster(x)) ||
      getLowerBounds(x, getCluster(x)) <
          computeDcDist(x, clusters.size() - 1)) {
    return getCluster(x);
  }
  return clusters.size() - 1;
}

template <class T>
inline size_t PKMeans<T>::getCluster(size_t x) {
  return clusterMap[x];
}

template <class T>
inline void PKMeans<T>::assignDistributions() {
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
inline void PKMeans<T>::computeClusterMean(size_t c) {
  if (clusterAssignments[c].size() == 0) return;
  newClusters[c].fill(0.0);
  for (size_t i = 0; i < clusterAssignments[c].size(); i++) {
    newClusters[c] += distributions[clusterAssignments[c][i]];
  }
  newClusters[c] /= float(clusterAssignments[c].size());
}

template <class T>
inline void PKMeans<T>::computeNewClusters() {
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
inline float PKMeans<T>::calcObjFn() {
  float sum = 0.0;
  for (size_t x = 0; x < distributions.size(); x++) {
    sum += PKMeans<float>::calcDist(distributions[x], clusters[getCluster(x)]);
  }
  return sum;
}

template <class T>
inline void PKMeans<T>::initLowerBounds() {
  lowerBounds.clear();
  lowerBounds.resize(distributions.size() * numClusters, 0);
}

template <class T>
inline void PKMeans<T>::initUpperBounds() {
  upperBounds.clear();
  for (size_t x = 0; x < distributions.size(); x++) {
    upperBounds.emplace_back(computeDcDist(x, getCluster(x)));
  }
}

template <class T>
inline void PKMeans<T>::initAssignments() {
  if (threads.size() > 1) {
    runThreads(distributions.size(), PKMeans::initAssignmentsThread);
  } else {
    if (clusters.size() == 1) {
      for (size_t x = 0; x < distributions.size(); x++) {
        computeDcDist(x, 0);
        clusterMap[x] = 0;
      }
      return;
    }
    for (size_t x = 0; x < distributions.size(); x++) {
      clusterMap[x] = findClosestInitCluster(x);
    }
  }
}

template <class T>
void *PKMeans<T>::initAssignmentsThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  if (pkmeans->clusters.size() == 1) {
    for (size_t x = threadArgs->start; x < threadArgs->end; x++) {
      pkmeans->computeDcDist(x, 0);
      pkmeans->clusterMap[x] = 0;
    }
  } else {
    for (size_t x = threadArgs->start; x < threadArgs->end; x++) {
      pkmeans->clusterMap[x] = pkmeans->findClosestInitCluster(x);
    }
  }
  pthread_exit(NULL);
}

template <class T>
inline void PKMeans<T>::initClusterDists() {
  clusterDists.clear();
  for (size_t c1 = 0; c1 < clusters.size(); c1++) {
    clusterDists.emplace_back();
    for (size_t c2 = 0; c2 < clusters.size(); c2++)
      clusterDists[c1].emplace_back(
          PKMeans<T>::emd<float>(clusters[c1], clusters[c2], denom));
  }
}

template <class T>
inline void PKMeans<T>::initSDists() {
  sDists.clear();
  for (size_t c = 0; c < clusters.size(); c++) {
    sDists.emplace_back();
  }
}

template <class T>
inline void PKMeans<T>::pushClusterDist() {
  size_t cNew = clusters.size() - 1;
  clusterDists.emplace_back();
  for (size_t c = 0; c < cNew; c++) {
    clusterDists[c].emplace_back(
        0.5f * PKMeans<T>::emd<float>(clusters[c], clusters[cNew], denom));
    clusterDists[cNew].emplace_back(clusterDists[c][cNew]);
  }
  clusterDists[cNew].emplace_back(0);
}

template <class T>
inline void PKMeans<T>::pushSDist() {
  sDists.emplace_back();
}

template <class T>
inline void PKMeans<T>::pushLowerBound() {
  for (size_t x = 0; x < distributions.size(); x++) {
    getLowerBounds(x, clusters.size() - 1) = 0;
  }
}

template <class T>
inline void PKMeans<T>::pushCluster(size_t x) {
  clusters.emplace_back(distributions[x]);
  clusterAssignments.emplace_back();
  newClusterDists.emplace_back(0);
  pushClusterDist();
  pushSDist();
  initAssignments();
}

template <class T>
inline void PKMeans<T>::computeClusterDists() {
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
inline void PKMeans<T>::computeLowerBounds() {
  if (threads.size() > 1) {
    runThreads(distributions.size(), PKMeans::computeLowerBoundsThread);
  } else {
    for (size_t x = 0; x < distributions.size(); x++)
      for (size_t c = 0; c < clusters.size(); c++)
        getLowerBounds(x, c) =
            fmax(getLowerBounds(x, c) - newClusterDists[c], 0);
  }
}

template <class T>
void *PKMeans<T>::computeLowerBoundsThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  for (size_t x = threadArgs->start; x < threadArgs->end; x++)
    for (size_t c = 0; c < pkmeans->clusters.size(); c++)
      pkmeans->getLowerBounds(x, c) =
          fmax(pkmeans->getLowerBounds(x, c) - pkmeans->newClusterDists[c], 0);
  pthread_exit(NULL);
}

template <class T>
inline void PKMeans<T>::computeUpperBounds() {
  if (threads.size() > 1) {
    runThreads(clusters.size(), PKMeans::computeUpperBoundsThread);
  } else {
    for (size_t c = 0; c < clusters.size(); c++) {
      newClusterDists[c] =
          PKMeans<T>::emd<float>(clusters[c], newClusters[c], denom);
      for (size_t i = 0; i < clusterAssignments[c].size(); i++) {
        upperBounds[clusterAssignments[c][i]] += newClusterDists[c];
      }
    }
  }
}

template <class T>
void *PKMeans<T>::computeUpperBoundsThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  for (size_t c = threadArgs->start; c < threadArgs->end; c++) {
    pkmeans->newClusterDists[c] = PKMeans<T>::emd<float>(
        pkmeans->clusters[c], pkmeans->newClusters[c], pkmeans->denom);
    for (size_t i = 0; i < pkmeans->clusterAssignments[c].size(); i++) {
      pkmeans->upperBounds[pkmeans->clusterAssignments[c][i]] +=
          pkmeans->newClusterDists[c];
    }
  }
  pthread_exit(NULL);
}

template <class T>
inline void PKMeans<T>::assignNewClusters() {
  for (size_t c = 0; c < clusters.size(); c++) clusters[c] = newClusters[c];
}

template <class T>
inline T PKMeans<T>::computeDcDist(size_t x, size_t c) {
  getLowerBounds(x, c) =
      PKMeans<T>::emd<float>(distributions[x], clusters[c], denom);
  return getLowerBounds(x, c);
}

template <class T>
inline T PKMeans<T>::cDist(size_t c1, size_t c2) {
  return clusterDists[c1][c2];
}

template <class T>
inline bool PKMeans<T>::needsClusterUpdateApprox(size_t x, size_t c) {
  return upperBounds[x] > getLowerBounds(x, c) &&
         upperBounds[x] > 0.5 * cDist(getCluster(x), c) && c != getCluster(x);
}

template <class T>
inline bool PKMeans<T>::needsClusterUpdate(size_t x, size_t c) {
  return computeDcDist(x, c) < upperBounds[x];
}

template <class T>
inline T &PKMeans<T>::getLowerBounds(size_t x, size_t c) {
  // printf("lowerBounds[%ld]\n", x * numClusters + c);
  return lowerBounds[x * numClusters + c];
}

namespace pkmeans {
template class PKMeans<float>;
template class PKMeans<std::uint8_t>;
}  // namespace pkmeans
