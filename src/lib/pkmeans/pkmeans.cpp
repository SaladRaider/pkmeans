#include <string>
#include <fstream>
#include <unordered_set>
#include <tgmath.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include "pkmeans.h"

using namespace pkmeans;

void PKMeans::run (int numClusters, int numThreads,
                   std::string inFilename,
                   std::string assignmentsOut,
                   std::string clustersOut) {
    printf ("Running pkmeans with args (k=%d, t=%d, i=%s, a=%s, c=%s)\n",
            numClusters, numThreads, inFilename.c_str (),
            assignmentsOut.c_str (), clustersOut.c_str());

    unsigned int numIterations = 1;
    initThreads (numThreads);
    readDistributions (inFilename);
    printf ("done reading file.\n");
    initClusters (numClusters);
    printf ("done intializing centroids.\n");
    initNewClusters ();
    initUpperBoundNeedsUpdate ();
    initUpperBounds ();
    printf ("done intializing.\n");
    computeNewClusters ();
    computeLowerBounds ();
    computeUpperBounds ();
    resetUpperBoundNeedsUpdate ();
    assignNewClusters ();
    printf ("Error is %f\n", calcObjFn ());
    while (!converged) {
        computeClusterDists ();
        assignDistributions ();
        computeNewClusters ();
        computeLowerBounds ();
        computeUpperBounds ();
        resetUpperBoundNeedsUpdate ();
        assignNewClusters ();
        printf ("Error is %f\n", calcObjFn ());
        numIterations += 1;
    }
    saveAssignments (assignmentsOut);
    saveClusters (clustersOut);
    pthread_attr_destroy (&threadAttr);

    printf ("pkmeans finished running with %u iterations.\n", numIterations);
}

void PKMeans::initThreads (int numThreads) {
    for (size_t i = 0; i < numThreads; i++) {
        threads.emplace_back ();
        threadArgs.emplace_back ((void*) this, 0, 0);
    }
    pthread_attr_init (&threadAttr);
    pthread_attr_setdetachstate (&threadAttr, PTHREAD_CREATE_JOINABLE);
}

void PKMeans::startThread (size_t tid, void* (*fn)(void*)) {
    int rc = pthread_create (&threads[tid], &threadAttr, fn,
                             (void*) &(threadArgs[tid]));
    if (rc) {
        std::cout << "Error: unable to create thread," << rc << '\n';
        exit (-1);
    }
}

void PKMeans::runThreads (size_t size, void* (*fn)(void*)) {
    size_t itemsPerThread = size / threads.size ();
    size_t i = 0;
    size_t tid = 0;
    for (tid = 0; tid < threads.size () - 1; tid++) {
        threadArgs[tid].start = i;
        i += itemsPerThread;
        threadArgs[tid].end = i;
        startThread (tid, fn);
    }
    threadArgs[tid].start = i;
    threadArgs[tid].end = size;
    startThread (tid, fn);
    joinThreads ();
}

void PKMeans::joinThreads () {
    int rc;
    void *status;
    for (size_t tid = 0; tid < threads.size (); tid++) {
        rc = pthread_join (threads[tid], &status);
        if (rc) {
            std::cout << "Error: unable to join," << rc << '\n';
            exit (-1);
        }
    }
}

void PKMeans::readDistributions (std::string inFilename) {
    const auto BUFFER_SIZE = 16*1024;
    const auto STR_SIZE = 64;
    int fd = open (inFilename.c_str (), O_RDONLY);
    if (fd == -1) {
        fprintf (stderr, "error opening file %s\n", inFilename.c_str ());
        return;
    }

    char buf[BUFFER_SIZE + 1];
    char floatStr[STR_SIZE];
    char *p;
    char *w;
    float bucketVal;
    Distribution<float> newDistribution;

    while(size_t bytes_read = read (fd, buf, BUFFER_SIZE)) {
        if(bytes_read == size_t (-1))
            fprintf (stderr, "read failed on file %s\n", inFilename.c_str ());
        if (!bytes_read)
            break;
        p = buf;
        w = &floatStr[0];
        memset (floatStr, '\0', sizeof (char) * STR_SIZE);
        for (p = buf; p < buf + bytes_read; ++p) {
            if (*p == '\n') {
                bucketVal = atof (floatStr);
                newDistribution.emplace_back (bucketVal);
                distributions.emplace_back (newDistribution);
                clusterMap.emplace_back (size_t (0));
                newDistribution.clear ();
                memset (floatStr, '\0', sizeof (char) * STR_SIZE);
                w = &floatStr[0];
            } else if (*p == ' ') {
                bucketVal = atof (floatStr);
                newDistribution.emplace_back (bucketVal);
                memset (floatStr, '\0', sizeof (char) * STR_SIZE);
                w = &floatStr[0];
            } else {
                memcpy (w++, p, sizeof (char) * 1);
            }
        }
    }
    if (newDistribution.size () != 0) {
        distributions.emplace_back (newDistribution);
        clusterMap.emplace_back (size_t (0));
    }
    denom = (distributions[0].size () - 1) * distributions[0].sum ();
}

void PKMeans::saveClusters (std::string outFilename) {
    std::ofstream f;
    f.open (outFilename);
    for (size_t c = 0; c < clusters.size (); c++) {
        f << clusters[c] << '\n';
    }
    f.close ();
}

void PKMeans::saveAssignments (std::string outFilename) {
    std::ofstream f;
    f.open (outFilename);
    for (size_t c = 0; c < clusterAssignments.size (); c++) {
        for (size_t j = 0; j < clusterAssignments[c].size (); j++) {
            f << c << ' ' << clusterAssignments[c][j] << '\n';
        }
    }
    f.close ();
}

void PKMeans::initClusters (int numClusters) {
    size_t kMax = size_t (numClusters);
    size_t x = size_t (rand () % int(distributions.size ()));
    float p;
    float pSum;
    Distribution<float> weightedP;

    weightedP.fill (0.0, distributions.size ());
    initLowerBounds ();

    // pick random 1st cluster
    pushCluster (x);

    for (size_t k = 1; k < kMax; k++) {
        // calculate weighted probabillities
        for (x = 0; x < distributions.size (); x++)
            weightedP[x] = lowerBounds[x][getCluster (x)];
        weightedP *= weightedP;
        weightedP /= weightedP.sum ();

        // select new cluster based on weighted probabillites
        p = float (rand ()) / float (RAND_MAX);
        pSum = 0.0;
        for (x = 0; x < distributions.size (); x++) {
            pSum += weightedP[x];
            if (pSum >= p) {
                pushCluster (x);
                break;
            }
        }
    }

    for (size_t x = 0; x < distributions.size (); x++) {
        clusterAssignments[getCluster (x)].emplace_back (x);
    }
}

void PKMeans::initNewClusters () {
    for (size_t c = 0; c < clusters.size (); c++)
        newClusters.emplace_back (clusters[c]);
}

void PKMeans::clearClusterAssignments () {
    for (size_t c = 0; c < clusterAssignments.size (); c++) {
        clusterAssignments[c].clear ();
    }
}

size_t PKMeans::findClosestCluster (size_t x) {
    if (upperBounds[x] <= sDists[getCluster (x)])
        return getCluster (x);
    for (size_t c = 0; c < clusters.size (); c++) {
        if (needsClusterUpdateApprox (x, c) &&
            needsClusterUpdate (x, c)) {
            converged = false;
            clusterMap[x] = c;
        }
    }
    return getCluster (x);
}

size_t PKMeans::findClosestInitCluster (size_t x) {
    // if 1/2 d(c,c') >= d(x,c), then
    //     d(x,c') >= d(x,c)
    if (clusters.size () == 1) {
        computeDcDist (x, 0);
        return 0;
    }
    size_t cx = getCluster (x);
    if (0.5 * cDist (cx, clusters.size () - 1) >= lowerBounds[x][cx]
        || lowerBounds[x][cx] < computeDcDist (x, clusters.size () - 1)) {
        return cx;
    }
    return clusters.size () - 1;
}

size_t PKMeans::getCluster (size_t x) {
    return clusterMap[x];
}

void PKMeans::assignDistributions () {
    converged = true;
    clearClusterAssignments ();
    if (threads.size () > 1) {
        runThreads (distributions.size (),
                    PKMeans::assignDistributionsThread);
    } else {
        for (size_t x = 0; x < distributions.size (); x++) {
            findClosestCluster (x);
        }
    }
    for (size_t x = 0; x < distributions.size (); x++) {
        clusterAssignments[getCluster (x)].emplace_back (x);
    }
}

void* PKMeans::assignDistributionsThread (void *args) {
    AssignThreadArgs *threadArgs = (AssignThreadArgs*) args;
    PKMeans *pkmeans = (PKMeans*) threadArgs->_this;
    for (size_t x = threadArgs->start; x < threadArgs->end; x++) {
        pkmeans->findClosestCluster (x);
    }
    pthread_exit (NULL);
}

void PKMeans::computeClusterMean (size_t c) {
    newClusters[c].fill (0.0);
    for (size_t i = 0; i < clusterAssignments[c].size (); i++) {
        newClusters[c] += distributions [clusterAssignments [c][i]];
    }
    newClusters[c] /= float(clusterAssignments[c].size ());
}

void PKMeans::computeNewClusters () {
    if (threads.size () > 1) {
        runThreads (clusters.size (),
                    PKMeans::computeNewClustersThread);
    } else {
        for (size_t c = 0; c < clusters.size (); c++) {
            computeClusterMean (c);
        }
    }
}

void* PKMeans::computeNewClustersThread (void *args) {
    AssignThreadArgs *threadArgs = (AssignThreadArgs*) args;
    PKMeans *pkmeans = (PKMeans*) threadArgs->_this;
    for (size_t c = threadArgs->start; c < threadArgs->end; c++) {
        pkmeans->computeClusterMean (c);
    }
    pthread_exit (NULL);
}

float PKMeans::calcObjFn () {
    float sum = 0.0;
    for (size_t x = 0; x < distributions.size (); x++) {
        sum += Distribution<float>::emd (distributions[x], clusters[getCluster (x)]);
    }
    return sum;
}

void PKMeans::initLowerBounds () {
    lowerBounds.clear ();
    for (size_t x = 0; x < distributions.size (); x++) {
        lowerBounds.emplace_back ();
        for (size_t c = 0; c < clusters.size (); c++) {
            lowerBounds[x].emplace_back (0.0);
        }
    }
}

void PKMeans::initUpperBounds () {
    upperBounds.clear ();
    for (size_t x = 0; x < distributions.size (); x++) {
        upperBounds.emplace_back (computeDcDist (x, getCluster (x)));
    }
}

void PKMeans::initAssignments () {
    for (size_t x = 0; x < distributions.size (); x++) {
        clusterMap[x] = findClosestInitCluster (x);
    }
}

void PKMeans::initClusterDists () {
    clusterDists.clear ();
    for (size_t c1 = 0; c1 < clusters.size (); c1++) {
        clusterDists.emplace_back ();
        for (size_t c2 = 0; c2 < clusters.size (); c2++)
            clusterDists[c1].emplace_back (
                Distribution<float>::emd8 (
                    clusters[c1], clusters[c2], denom
            ));
    }
}

void PKMeans::initSDists () {
    sDists.clear ();
    for (size_t c = 0; c < clusters.size (); c++) {
        sDists.emplace_back ();
    }
}

void PKMeans::initUpperBoundNeedsUpdate () {
    upperBoundNeedsUpdate.clear ();
    for (size_t x = 0; x < distributions.size (); x++) {
        upperBoundNeedsUpdate.emplace_back (true);
    }
}

void PKMeans::pushClusterDist () {
    size_t cNew = clusters.size () - 1;
    for (size_t c = 0; c < cNew; c++) {
        clusterDists[c].emplace_back (Distribution<float>::emd8 (
                clusters[c], clusters[cNew], denom
        ));
    }
    clusterDists.emplace_back ();
    for (size_t c = 0; c < clusters.size (); c++) {
        clusterDists[cNew].emplace_back (Distribution<float>::emd8 (
                clusters[cNew], clusters[c], denom
        ));
    }
}

void PKMeans::pushSDist () {
    sDists.emplace_back ();
}

void PKMeans::pushLowerBound () {
    for (size_t x = 0; x < distributions.size (); x++) {
        lowerBounds[x].emplace_back (0.0);
    }
}

void PKMeans::pushCluster (size_t x) {
    clusters.emplace_back (distributions[x]);
    clusterAssignments.emplace_back ();
    pushClusterDist ();
    pushSDist ();
    pushLowerBound ();
    initAssignments ();
}

void PKMeans::resetUpperBoundNeedsUpdate () {
    for (size_t x = 0; x < distributions.size (); x++) {
        upperBoundNeedsUpdate[x] = true;
    }
}

void PKMeans::computeClusterDists () {
    for (size_t c1 = 0; c1 < clusters.size (); c1++) {
        sDists[c1] = 255;
        for (size_t c2 = 0; c2 < clusters.size (); c2++) {
            clusterDists[c1][c2] = Distribution<float>::emd8 (
                clusters[c1], clusters[c2], denom
            );
            if (c1 != c2 && 0.5 * clusterDists[c1][c2] < sDists[c1])
                sDists[c1] = 0.5 * clusterDists[c1][c2];
        }
    }
}

void PKMeans::computeLowerBounds () {
    for (size_t c = 0; c < clusters.size (); c++)
    for (size_t x = 0; x < distributions.size (); x++)
        lowerBounds[x][c] = fmax (
            lowerBounds[x][c] - Distribution<float>::emd8 (
                clusters[c], newClusters[c], denom
            ), 0
        );
}

void PKMeans::computeUpperBounds () {
    for (size_t x = 0; x < distributions.size (); x++)
        upperBounds[x] += Distribution<float>::emd8 (
            clusters[getCluster (x)],
            newClusters[getCluster (x)],
            denom
        );
}

void PKMeans::assignNewClusters () {
    for (size_t c = 0; c < clusters.size (); c++)
        clusters[c] = newClusters[c];
}

std::uint8_t PKMeans::computeDcDist (size_t x, size_t c) {
    lowerBounds[x][c] = Distribution<float>::emd8 (
        distributions[x], clusters[c], denom
    );
    return lowerBounds[x][c];
}

std::uint8_t PKMeans::cDist (size_t c1, size_t c2) {
    return clusterDists[c1][c2];
}

bool PKMeans::needsClusterUpdateApprox (size_t x, size_t c) {
    return c != getCluster (x) &&
        upperBounds[x] > lowerBounds[x][c] &&
        upperBounds[x] > 0.5 * cDist (getCluster (x), c);
}

bool PKMeans::needsClusterUpdate (size_t x, size_t c) {
    size_t cx = getCluster (x);
    size_t dcDist;
    if (upperBoundNeedsUpdate[x]) {
        dcDist = computeDcDist (x, cx);
        upperBounds[x] = dcDist;
        upperBoundNeedsUpdate[x] = false;
    } else {
        dcDist = upperBounds[x];
    }
    return (dcDist > lowerBounds[x][c] ||
            dcDist > 0.5 * cDist (cx, c)) &&
           computeDcDist (x, c) < dcDist;
}
