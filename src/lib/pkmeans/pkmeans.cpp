#include <string>
#include <fstream>
#include <unordered_set>
#include <tgmath.h>
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
    readDistributions (inFilename);
    initClusters (numClusters);
    initNewClusters ();
    initLowerBounds ();
    initClusterDistributionDists ();
    initClusterDists ();
    initSDists ();
    initR ();
    initAssignments ();
    initUpperBounds ();
    printf ("Error is %f\n", calcObjFn ());
    while (!converged) {
        computeClusterDists ();
        assignDistributions ();
        computeNewClusters ();
        computeLowerBounds ();
        computeUpperBounds ();
        resetR ();
        assignNewClusters ();
        printf ("Error is %f\n", calcObjFn ());
        numIterations += 1;
    }
    saveAssignments (assignmentsOut);
    saveClusters (clustersOut);

    printf ("pkmeans finished running with %u iterations.\n", numIterations);
}

void PKMeans::readDistributions (std::string inFilename) {
    std::ifstream infile;
    std::string fileBuffer;
    infile.open (inFilename);
    while (infile) {
        Distribution<double> newDistribution;
        if (!(infile >> newDistribution))
            break;
        distributions.emplace_back (newDistribution);
        prevClusterAssignments.emplace_back (size_t (-1));
    }
    infile.close ();
}

void PKMeans::saveClusters (std::string outFilename) {
    std::ofstream f;
    f.open (outFilename);
    for (size_t i = 0; i < clusters.size (); i++) {
        f << clusters[i] << '\n';
    }
    f.close ();
}

void PKMeans::saveAssignments (std::string outFilename) {
    std::ofstream f;
    f.open (outFilename);
    for (size_t i = 0; i < clusterAssignments.size (); i++) {
        for (size_t j = 0; j < clusterAssignments[i].size (); j++) {
            f << i << ' ' << clusterAssignments[i][j] << '\n';
        }
    }
    f.close ();
}

void PKMeans::initClusters (int numClusters) {
    size_t kMax = size_t (numClusters);
    size_t x = size_t (rand () % int(distributions.size ()));
    double p;
    double pSum;
    Distribution<double> weightedP;

    weightedP.fill (0.0, distributions.size ());

    // pick random 1st cluster
    clusters.emplace_back (distributions[x]);
    clusterAssignments.emplace_back ();

    for (size_t k = 1; k < kMax; k++) {
        initAssignments ();

        // calculate weighted probabillities
        for (size_t c = 0; c < clusterAssignments.size (); c++) {
            for (size_t i = 0; i < clusterAssignments[i].size (); i++) {
                x = clusterAssignments [c][i];
                weightedP[x] = Distribution<double>::emd (
                    distributions[x], clusters[c]
                );
            }
        }
        weightedP *= weightedP;
        weightedP /= weightedP.sum ();

        // select new cluster based on weighted probabillites
        p = double (rand ()) / double (RAND_MAX);
        pSum = 0.0;
        for (x = 0; x < distributions.size (); x++) {
            pSum += weightedP[x];
            if (pSum >= p) {
                clusters.emplace_back (distributions[x]);
                clusterAssignments.emplace_back ();
                break;
            }
        }
    }
}

void PKMeans::initNewClusters () {
    for (size_t c = 0; c < clusters.size (); c++)
        newClusters.emplace_back (clusters[c]);
}

void PKMeans::clearClusterAssignments () {
    for (size_t i = 0; i < clusterAssignments.size (); i++) {
        clusterAssignments[i].clear ();
    }
}

size_t PKMeans::findClosestCluster (size_t x) {
    size_t cx = getCluster (x);
    if (upperBounds[x] <= sDists[cx])
        return cx;
    for (size_t c = 0; c < clusters.size (); c++) {
        if (!needsClusterUpdate (x, c))
            continue;
        if (r[x]) {
            computeDcDist (x, cx);
            r[x] = false;
        } else {
            clusterDistributionDists[x][c] = upperBounds[x];
        }
        if (dcDist (x, cx) > lowerBounds[x][c] ||
            dcDist (x, cx) > 0.5 * cDist (cx, c)) {
            if (computeDcDist (x, c) < dcDist (x, cx)) {
                cx = c;
            }
        }
    }
    return cx;
}

size_t PKMeans::findClosestInitCluster (size_t distributionIdx) {
    // if 1/2 d(c,c') >= d(x,c), then
    //     d(x,c') >= d(x,c)
    size_t closestIdx = 0;
    double dist = computeDcDist (distributionIdx, 0);
    double _cDist;
    double newDist;
    for (size_t i = 1; i < clusters.size (); i++) {
        _cDist = cDist (i-1, i);
        if (0.5 * _cDist >= dist)
            continue;
        newDist = computeDcDist (distributionIdx, i);
        if (newDist < dist) {
            closestIdx = i;
            dist = newDist;
        }
    }
    return closestIdx;
}

size_t PKMeans::getCluster (size_t distributionIdx) {
    return prevClusterAssignments[distributionIdx];
}

void PKMeans::assignDistributions () {
    converged = true;
    clearClusterAssignments ();
    size_t closestClusterIdx;
    for (size_t x = 0; x < distributions.size (); x++) {
        closestClusterIdx = findClosestCluster (x);
        clusterAssignments[closestClusterIdx].emplace_back (x);
        if (closestClusterIdx != prevClusterAssignments[x]) {
            converged = false;
            prevClusterAssignments[x] = closestClusterIdx;
        }
    }
}

void PKMeans::computeClusterMean (size_t idx) {
    newClusters[idx].fill (0.0);
    for (size_t i = 0; i < clusterAssignments[idx].size (); i++) {
        newClusters[idx] += distributions [clusterAssignments [idx][i]];
    }
    newClusters[idx] /= double(clusterAssignments[idx].size ());
}

void PKMeans::computeNewClusters () {
    for (size_t i = 0; i < clusters.size (); i++) {
        computeClusterMean (i);
    }
}

double PKMeans::calcObjFn () {
    double sum = 0.0;
    for (size_t x = 0; x < distributions.size (); x++) {
        sum += dcDist (x, getCluster (x));
    }
    return sum;
}

void PKMeans::initLowerBounds () {
    for (size_t i = 0; i < distributions.size (); i++) {
        lowerBounds.emplace_back ();
        for (size_t j = 0; j < clusters.size (); j++) {
            lowerBounds[i].emplace_back ();
        }
    }
}

void PKMeans::initUpperBounds () {
    for (size_t x = 0; x < distributions.size (); x++) {
        upperBounds.emplace_back (dcDist (x, getCluster (x)));
    }
}

void PKMeans::initAssignments () {
    size_t closestClusterIdx;
    clearClusterAssignments ();
    for (size_t i = 0; i < distributions.size (); i++) {
        closestClusterIdx = findClosestInitCluster (i);
        clusterAssignments[closestClusterIdx].emplace_back (i);
        prevClusterAssignments[i] = closestClusterIdx;
    }
}

void PKMeans::initClusterDistributionDists () {
    for (size_t i = 0; i < distributions.size (); i++) {
        clusterDistributionDists.emplace_back ();
        for (size_t j = 0; j < clusters.size (); j++)
            clusterDistributionDists.emplace_back ();
    }
}

void PKMeans::initClusterDists () {
    for (size_t i = 0; i < clusters.size (); i++) {
        clusterDists.emplace_back ();
        for (size_t j = 0; j < clusters.size (); j++)
            clusterDists.emplace_back ();
    }
}

void PKMeans::initSDists () {
    for (size_t i = 0; i < clusters.size (); i++) {
        sDists.emplace_back ();
    }
}

void PKMeans::initR () {
    for (size_t i = 0; i < clusters.size (); i++) {
        r.emplace_back (true);
    }
}

void PKMeans::resetR () {
    for (size_t i = 0; i < clusters.size (); i++) {
        r[i] = true;
    }
}

void PKMeans::computeClusterDists () {
    for (size_t i = 0; i < clusters.size (); i++) {
        sDists[i] = DBL_MAX;
        for (size_t j = 0; j < clusters.size (); j++) {
            clusterDists[i][j] = Distribution<double>::emd (
                clusters[i], clusters[j]
            );
            if (clusterDists[i][j] < sDists[i])
                sDists[i] = clusterDists[i][j];
        }
    }
}

void PKMeans::computeLowerBounds () {
    for (size_t x = 0; x < distributions.size (); x++)
    for (size_t c = 0; c < distributions.size (); c++)
        lowerBounds[x][c] = fmax (
            lowerBounds[x][c] - Distribution<double>::emd (
                clusters[c], newClusters[c]
            ), 0
        );
}

void PKMeans::computeUpperBounds () {
    for (size_t x = 0; x < distributions.size (); x++)
        upperBounds[x] += Distribution<double>::emd (
            clusters[getCluster (x)],
            newClusters[getCluster (x)]
        );
}

void PKMeans::assignNewClusters () {
    for (size_t c = 0; c < distributions.size (); c++)
        clusters[c] = newClusters[c];
}

double PKMeans::computeDcDist (size_t x, size_t c) {
    clusterDistributionDists[x][c] = Distribution<double>::emd (
        distributions[x], clusters[c]
    );
    lowerBounds[x][c] = clusterDistributionDists[x][c];
    return clusterDistributionDists[x][c];
}

double PKMeans::dcDist (size_t x, size_t c) {
    return clusterDistributionDists[x][c];
}

double PKMeans::cDist (size_t c1, size_t c2) {
    return clusterDists[c1][c2];
}

bool PKMeans::needsClusterUpdate (size_t x, size_t c) {
    return c != getCluster (x) &&
        upperBounds[x] > lowerBounds[x][c] &&
        upperBounds[x] > 0.5 * cDist (getCluster (x), c);
}

