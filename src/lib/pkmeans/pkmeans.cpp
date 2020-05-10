#include <string>
#include <fstream>
#include <unordered_set>
#include <tgmath.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
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

    printf ("pkmeans finished running with %u iterations.\n", numIterations);
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
    char doubleStr[STR_SIZE];
    char *p;
    char *w;
    double bucketVal;
    Distribution<double> newDistribution;

    while(size_t bytes_read = read (fd, buf, BUFFER_SIZE)) {
        if(bytes_read == size_t (-1))
            fprintf (stderr, "read failed on file %s\n", inFilename.c_str ());
        if (!bytes_read)
            break;
        p = buf;
        w = &doubleStr[0];
        memset (doubleStr, '\0', sizeof (char) * STR_SIZE);
        for (p = buf; p < buf + bytes_read; ++p) {
            if (*p == '\n') {
                bucketVal = atof (doubleStr);
                newDistribution.emplace_back (bucketVal);
                distributions.emplace_back (newDistribution);
                clusterMap.emplace_back (size_t (0));
                newDistribution.clear ();
                memset (doubleStr, '\0', sizeof (char) * STR_SIZE);
                w = &doubleStr[0];
            } else if (*p == ' ') {
                bucketVal = atof (doubleStr);
                newDistribution.emplace_back (bucketVal);
                memset (doubleStr, '\0', sizeof (char) * STR_SIZE);
                w = &doubleStr[0];
            } else {
                memcpy (w++, p, sizeof (char) * 1);
            }
        }
    }
    if (newDistribution.size () != 0) {
        distributions.emplace_back (newDistribution);
        clusterMap.emplace_back (size_t (0));
    }
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
        p = double (rand ()) / double (RAND_MAX);
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
    size_t cx;
    for (size_t x = 0; x < distributions.size (); x++) {
        cx = findClosestCluster (x);
        clusterAssignments[cx].emplace_back (x);
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
                Distribution<double>::emd (
                    clusters[c1], clusters[c2]
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
        clusterDists[c].emplace_back (Distribution<double>::emd (
                clusters[c], clusters[cNew]
        ));
    }
    clusterDists.emplace_back ();
    for (size_t c = 0; c < clusters.size (); c++) {
        clusterDists[cNew].emplace_back (Distribution<double>::emd (
                clusters[cNew], clusters[c]
        ));
    }
}

void PKMeans::pushSDist () {
    sDists.emplace_back ();
}

void PKMeans::pushLowerBound () {
    for (size_t x = 0; x < distributions.size (); x++) {
        lowerBounds[x].emplace_back ();
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
        sDists[c1] = DBL_MAX;
        for (size_t c2 = 0; c2 < clusters.size (); c2++) {
            clusterDists[c1][c2] = Distribution<double>::emd (
                clusters[c1], clusters[c2]
            );
            if (c1 != c2 && 0.5 * clusterDists[c1][c2] < sDists[c1])
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
    lowerBounds[x][c] = Distribution<double>::emd (
        distributions[x], clusters[c]
    );
    return lowerBounds[x][c];
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
