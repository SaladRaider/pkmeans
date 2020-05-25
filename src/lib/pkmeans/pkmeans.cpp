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
                     float _maxMissingMass, size_t _seed, int backupEvery,
                     bool useTimeSeed, const std::string &inFilename,
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
  saveEvery = backupEvery;
  printf(
      "Running pkmeans with args (k=%ld, t=%d, p=%f, m=%f, l=%s, s=%ld, b=%d, "
      "i=%s, a=%s, c=%s, e=%s, q=%s)\n",
      numClusters, numThreads, _confidenceProb, _maxMissingMass,
      sizeof(T) == 1 ? "true" : "false", seed, saveEvery, inFilename.c_str(),
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

  unsigned int numIterations = 0;
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
  auto prevError = objError;
  if (!quiet) printf("Error is %f\n", objError);
  while (!converged) {
    if (saveEvery > 0 && numIterations % saveEvery == 0) {
      if (!quiet) printf("\033[1msaving...\033[0m\n");
      saveAssignments(assignmentsOut);
      saveClusters(clustersOut);
      if (!quiet) printf("\033[1mdone saving\033[0m\n");
    }
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
    objError = calcObjFn();
    if (!quiet) printf("Error is %f\n", objError);
    if (!quiet)
      printf("%f%% improvement\n", (prevError - objError) / prevError * 100.f);
    prevError = objError;
    numIterations += 1;
    if (!quiet) printf("done with itteration %u\n", numIterations);
  }
  if (!quiet) printf("calculating obj fn...\n");
  objError = calcObjFn();
  if (!quiet) printf("Error is %f\n", objError);
  if (objError < bestError) {
    bestError = objError;
    if (quiet)
      printf("\nFound new best error: %f; seed: %ld; saving assignments...\n",
             bestError, seed);
    else
      printf("Found new best error: %f; seed: %ld; saving assignments...\n",
             bestError, seed);
    saveAssignments(assignmentsOut);
    saveClusters(clustersOut);
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
void PKMeans<T>::markClustersObserved(double objError) {
  size_t h = std::hash<double>()(objError);
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
  // TOOD: get rid of cluster permutation info
  return boost::hash_range(clusterMap.begin(), clusterMap.end());
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
  clusterSize.clear();
  newClusters.clear();
  newClusterDists.clear();
  if (threads.size() > 1) {
    runThreads(distributions.size(), PKMeans<T>::resetLowerBoundsThread);
  } else {
    for (size_t x = 0; x < distributions.size(); x++)
      for (size_t c = 0; c < numClusters; c++) getLowerBounds(x, c) = 0;
  }
  upperBounds.clear();
  clusterDists.clear();
  sDists.clear();
  converged = false;
  if (!quiet) printf("done resetting run variables\n");
}

template <class T>
void *PKMeans<T>::resetLowerBoundsThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  for (size_t x = threadArgs->start; x < threadArgs->end; x++)
    for (size_t c = 0; c < pkmeans->numClusters; c++)
      pkmeans->getLowerBounds(x, c) = 0;
  pthread_exit(NULL);
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
  constexpr auto BUFFER_SIZE = 4 * 1024 * 1024; // 4 MB
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
  memset(floatStr, '\0', sizeof(char) * STR_SIZE);
  while (size_t bytes_read = read(fd, buf, BUFFER_SIZE)) {
    if (bytes_read == size_t(-1))
      fprintf(stderr, "read failed on file %s\n", inFilename.c_str());
    if (!bytes_read) break;
    p = buf;
    for (p = buf; p < buf + bytes_read; ++p) {
      if (*p == '\n') {
        bucketVal = atof(floatStr);
        newDistribution.emplace_back(bucketVal);
        distributions.emplace_back(newDistribution);
        clusterMap.emplace_back(size_t(0));
        r.push_back(true);
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
    r.push_back(true);
  }
  printf("getting max distance...\n");
  float maxDist = 0.f;
  float dist;
  Distribution<float> mean = distributions[0];
  mean.fill(0.f);
  for (size_t x = 0; x < distributions.size(); x++) {
    mean += distributions[x];
  }
  mean /= float(distributions.size());
  for (size_t x = 0; x < distributions.size(); x++) {
    dist = PKMeans<float>::calcDist(mean, distributions[x]);
    if (dist > maxDist) {
      maxDist = dist;
    }
  }
  // actual max is 2*maxDist, but 1*maxDist keeps more precision
  denom = maxDist;
  printf("max distance: %f\n", maxDist);
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
  auto progressBar = ProgressBar<50>();
  constexpr int showEvery = 1024 * 1024 / (10 + 1 + 3 + 1);  // ~1 MB
  for (size_t x = 0; x < distributions.size(); x++) {
    f << x << ' ' << getCluster(x) << '\n';
    if (showEvery < distributions.size() && x % showEvery == 0)
      progressBar.show(float(x) / distributions.size());
  }
  f.close();
  std::cout << '\n';
}

template <class T>
void PKMeans<T>::initClusters() {
  size_t kMax = size_t(numClusters);
  size_t xx = size_t(rand() % int(distributions.size()));
  size_t newClusterIdx;
  double p;
  double weightedSum;
  weightedP.assign(distributions.size(), 0.0);
  ProgressBar<50> progressBar;
  if (!quiet) progressBar.show(0.f);

  // pick random 1st cluster
  pushCluster(xx);
  if (!quiet) progressBar.show(1.f / kMax);

  for (size_t k = 1; k < kMax; k++) {
    // calculate weighted probabillities
    weightedSum = 0;
    weightedSums.assign(threads.size(), 0.0);
    if (threads.size() > 1) {
      runThreads(distributions.size(), calcWeighted);
      for (auto i = 0; i < weightedSums.size(); i++) {
        weightedSum += weightedSums[i];
      }
      p = (double(rand()) / double(RAND_MAX)) * weightedSum;
      for (auto i = 0; i < weightedSums.size(); i++) {
        p -= weightedSums[i];
        if (p <= 0) {
          p += weightedSums[i];
          for (size_t x = i * (distributions.size() / threads.size());
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
        weightedP[x] = double(getLowerBounds(x, getCluster(x)));
        weightedP[x] *= weightedP[x];
        weightedSum += weightedP[x];
      }
      // select new cluster based on weighted probabillites
      p = (double(rand()) / double(RAND_MAX)) * weightedSum;
      for (size_t x = 0; x < distributions.size(); x++) {
        p -= weightedP[x];
        if (p <= 0) {
          pushCluster(x);
          break;
        }
      }
    }
    if (!quiet) progressBar.show((k + 1.f) / kMax);
  }
  weightedP.clear();
  weightedP.resize(0);
  if (!quiet) printf("\n");
}

template <class T>
void *PKMeans<T>::calcWeighted(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  size_t tid = size_t(threadArgs->start / (pkmeans->distributions.size() /
                                           pkmeans->threads.size()));
  for (size_t x = threadArgs->start; x < threadArgs->end; x++) {
    pkmeans->weightedP[x] =
        double(pkmeans->getLowerBounds(x, pkmeans->getCluster(x)));
    pkmeans->weightedP[x] *= pkmeans->weightedP[x];
    pkmeans->weightedSums[tid] += pkmeans->weightedP[x];
  }
  pthread_exit(NULL);
};

template <class T>
inline void PKMeans<T>::initNewClusters() {
  for (size_t tid = 0; tid < threads.size(); tid++) {
    newClusterSums.emplace_back(clusters.size(), clusters[0]);
    clusterSizes.emplace_back(clusters.size());
  }
  for (size_t c = 0; c < clusters.size(); c++) {
    newClusters.emplace_back(clusters[c]);
  }
}

template <class T>
inline size_t PKMeans<T>::findClosestCluster(size_t x) {
  if (upperBounds[x] <= sDists[getCluster(x)]) return getCluster(x);
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
  if (threads.size() > 1) {
    runThreads(distributions.size(), PKMeans::assignDistributionsThread);
  } else {
    for (size_t x = 0; x < distributions.size(); x++) {
      findClosestCluster(x);
    }
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
inline void PKMeans<T>::computeNewClusters() {
  for (size_t c = 0; c < clusters.size(); c++) {
    newClusters[c].fill(0.f);
    clusterSize[c] = 0.f;
  }
  if (threads.size() > 1) {
    runThreads(distributions.size(), PKMeans::computeNewClustersThread);
    for (size_t tid = 0; tid < threads.size(); tid++) {
      for (size_t c = 0; c < clusters.size(); c++) {
        newClusters[c] += newClusterSums[tid][c];
        clusterSize[c] += clusterSizes[tid][c];
      }
    }
  } else {
    for (size_t x = 0; x < distributions.size(); x++) {
      newClusters[getCluster(x)] += distributions[x];
      clusterSize[getCluster(x)] += 1;
    }
  }
  for (size_t c = 0; c < clusters.size(); c++) {
    newClusters[c] /= float(clusterSize[c]);
  }
}

template <class T>
void *PKMeans<T>::computeNewClustersThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  size_t tid = size_t(threadArgs->start / (pkmeans->distributions.size() /
                                           pkmeans->threads.size()));
  for (size_t c = 0; c < pkmeans->clusters.size(); c++) {
    pkmeans->newClusterSums[tid][c].fill(0.f);
    pkmeans->clusterSizes[tid][c] = 0.f;
  }
  for (size_t x = threadArgs->start; x < threadArgs->end; x++) {
    pkmeans->newClusterSums[tid][pkmeans->getCluster(x)] +=
        pkmeans->distributions[x];
    pkmeans->clusterSizes[tid][pkmeans->getCluster(x)] += 1;
  }
  pthread_exit(NULL);
}

template <class T>
inline double PKMeans<T>::calcObjFn() {
  double sum = 0.0;
  if (threads.size() > 1) {
    std::fill(weightedSums.begin(), weightedSums.end(), 0);
    runThreads(distributions.size(), PKMeans<T>::calcObjFnThread);
    for (size_t tid = 0; tid < weightedSums.size(); tid++) {
      sum += weightedSums[tid];
    }
  } else {
    double dist = 0.0;
    for (size_t x = 0; x < distributions.size(); x++) {
      dist =
          PKMeans<float>::calcDist(distributions[x], clusters[getCluster(x)]);
      sum += dist * dist;
    }
  }
  return sum;
}

template <class T>
void *PKMeans<T>::calcObjFnThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  double dist = 0.0;
  size_t tid = size_t(threadArgs->start / (pkmeans->distributions.size() /
                                           pkmeans->threads.size()));
  for (size_t x = threadArgs->start; x < threadArgs->end; x++) {
    dist = PKMeans<float>::calcDist(pkmeans->distributions[x],
                                    pkmeans->clusters[pkmeans->getCluster(x)]);
    pkmeans->weightedSums[tid] += dist * dist;
  }
  pthread_exit(NULL);
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
  clusterSize.emplace_back(0);
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
      sDists[c1] = std::numeric_limits<T>::max();
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
    pkmeans->sDists[c1] = std::numeric_limits<T>::max();
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
  for (size_t c = 0; c < clusters.size(); c++) {
    newClusterDists[c] =
        PKMeans<T>::emd<float>(clusters[c], newClusters[c], denom);
  }
  if (threads.size() > 1) {
    runThreads(distributions.size(), PKMeans::computeUpperBoundsThread);
  } else {
    for (size_t x = 0; x < distributions.size(); x++) {
      upperBounds[x] += newClusterDists[getCluster(x)];
      r[x] = true;
    }
  }
}

template <class T>
void *PKMeans<T>::computeUpperBoundsThread(void *args) {
  ThreadArgs *threadArgs = (ThreadArgs *)args;
  PKMeans *pkmeans = (PKMeans *)threadArgs->_this;
  for (size_t x = threadArgs->start; x < threadArgs->end; x++) {
    pkmeans->upperBounds[x] += pkmeans->newClusterDists[pkmeans->getCluster(x)];
    pkmeans->r[x] = true;
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
  if (r[x]) {
    upperBounds[x] = computeDcDist(x, getCluster(x));
    r[x] = false;
  }
  return (upperBounds[x] > getLowerBounds(x, c) ||
          upperBounds[x] > 0.5 * cDist(getCluster(x), c)) &&
         computeDcDist(x, c) < upperBounds[x];
}

template <class T>
inline T &PKMeans<T>::getLowerBounds(size_t x, size_t c) {
  return lowerBounds[x * numClusters + c];
}

namespace pkmeans {
template class PKMeans<float>;
template class PKMeans<std::uint8_t>;
}  // namespace pkmeans
