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
    double p;
    double pSum;
    Distribution<double> weightedP;

    weightedP.fill (0.0, distributions.size ());

    // pick random 1st cluster
    clusters.emplace_back (distributions[x]);
    clusterAssignments.emplace_back ();

    for (size_t k = 1; k < kMax; k++) {
        initClusterDists ();
        initClusterDistributionDists ();
        initSDists ();
        initLowerBounds ();
        computeClusterDists ();
        initAssignments ();

        // calculate weighted probabillities
        for (x = 0; x < distributions.size (); x++)
            weightedP[x] = computeDcDist (x, getCluster (x));
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
    for (size_t c = 0; c < clusterAssignments.size (); c++) {
        clusterAssignments[c].clear ();
    }
}

size_t PKMeans::findClosestCluster (size_t x) {
    size_t cx = getCluster (x);
    if (upperBounds[x] <= sDists[cx])
        return cx;
    for (size_t c = 0; c < clusters.size (); c++) {
        if (needsClusterUpdateApprox (x, c) &&
            needsClusterUpdate (x, c))
            cx = c;
    }
    return cx;
}

size_t PKMeans::findClosestInitCluster (size_t x) {
    // if 1/2 d(c,c') >= d(x,c), then
    //     d(x,c') >= d(x,c)
    size_t closestC = 0;
    double dist = computeDcDist (x, closestC);
    double _cDist;
    double newDist;
    for (size_t c = 1; c < clusters.size (); c++) {
        _cDist = cDist (closestC, c);
        if (0.5 * _cDist >= dist)
            continue;
        newDist = computeDcDist (x, c);
        if (newDist < dist) {
            closestC = c;
            dist = newDist;
        }
    }
    return closestC;
}

size_t PKMeans::getCluster (size_t distributionIdx) {
    return prevClusterAssignments[distributionIdx];
}

void PKMeans::assignDistributions () {
    converged = true;
    clearClusterAssignments ();
    size_t cx;
    for (size_t x = 0; x < distributions.size (); x++) {
        cx = findClosestCluster (x);
        clusterAssignments[cx].emplace_back (x);
        if (cx != prevClusterAssignments[x]) {
            converged = false;
            prevClusterAssignments[x] = cx;
        }
    }
}

void PKMeans::computeClusterMean (size_t c) {
    newClusters[c].fill (0.0);
    for (size_t i = 0; i < clusterAssignments[c].size (); i++) {
        newClusters[c] += distributions [clusterAssignments [c][i]];
    }
    newClusters[c] /= double(clusterAssignments[c].size ());
}

void PKMeans::computeNewClusters () {
    for (size_t c = 0; c < clusters.size (); c++) {
        computeClusterMean (c);
    }
}

double PKMeans::calcObjFn () {
    double sum = 0.0;
    for (size_t x = 0; x < distributions.size (); x++) {
        sum += computeDcDist (x, getCluster (x));
    }
    return sum;
}

void PKMeans::initLowerBounds () {
    lowerBounds.clear ();
    for (size_t x = 0; x < distributions.size (); x++) {
        lowerBounds.emplace_back ();
        for (size_t c = 0; c < clusters.size (); c++) {
            lowerBounds[x].emplace_back ();
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
    size_t closestClusterIdx;
    clearClusterAssignments ();
    for (size_t x = 0; x < distributions.size (); x++) {
        closestClusterIdx = findClosestInitCluster (x);
        clusterAssignments[closestClusterIdx].emplace_back (x);
        prevClusterAssignments[x] = closestClusterIdx;
    }
}

void PKMeans::initClusterDistributionDists () {
    clusterDistributionDists.clear ();
    for (size_t x = 0; x < distributions.size (); x++) {
        clusterDistributionDists.emplace_back ();
        for (size_t c = 0; c < clusters.size (); c++)
            clusterDistributionDists[x].emplace_back ();
    }
}

void PKMeans::initClusterDists () {
    clusterDists.clear ();
    for (size_t c1 = 0; c1 < clusters.size (); c1++) {
        clusterDists.emplace_back ();
        for (size_t c2 = 0; c2 < clusters.size (); c2++)
            clusterDists[c1].emplace_back ();
    }
}

void PKMeans::initSDists () {
    sDists.clear ();
    for (size_t c = 0; c < clusters.size (); c++) {
        sDists.emplace_back ();
    }
}

void PKMeans::initR () {
    r.clear ();
    for (size_t x = 0; x < distributions.size (); x++) {
        r.emplace_back (true);
    }
}

void PKMeans::resetR () {
    for (size_t x = 0; x < distributions.size (); x++) {
        r[x] = true;
    }
}

void PKMeans::computeClusterDists () {
    for (size_t c1 = 0; c1 < clusters.size (); c1++) {
        sDists[c1] = DBL_MAX;
        for (size_t c2 = 0; c2 < clusters.size (); c2++) {
            clusterDists[c1][c2] = Distribution<double>::emd (
                clusters[c1], clusters[c2]
            );
            if (c1 != c2 && clusterDists[c1][c2] < sDists[c1])
                sDists[c1] = 0.5 * clusterDists[c1][c2];
        }
    }
}

void PKMeans::computeLowerBounds () {
    for (size_t x = 0; x < distributions.size (); x++)
    for (size_t c = 0; c < clusters.size (); c++)
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
    for (size_t c = 0; c < clusters.size (); c++)
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

bool PKMeans::needsClusterUpdateApprox (size_t x, size_t c) {
    return c != getCluster (x) &&
        upperBounds[x] > lowerBounds[x][c] &&
        upperBounds[x] > 0.5 * cDist (getCluster (x), c);
}

bool PKMeans::needsClusterUpdate (size_t x, size_t c) {
    size_t cx = getCluster (x);
    if (r[x]) {
        computeDcDist (x, cx);
        r[x] = false;
    } else {
        clusterDistributionDists[x][c] = upperBounds[x];
    }
    return (dcDist (x, cx) > lowerBounds[x][c] ||
            dcDist (x, cx) > 0.5 * cDist (cx, c)) &&
            computeDcDist (x, c) < dcDist (x, cx);
}
