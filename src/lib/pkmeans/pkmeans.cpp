#include <string>
#include <fstream>
#include <unordered_set>
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
    assignDistributions ();
    printf ("Error is %f\n", calcObjFn ());
    computeNewClusters ();
    while (!converged) {
        assignDistributions ();
        printf ("Error is %f\n", calcObjFn ());
        computeNewClusters ();
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
    size_t distIdx = size_t (rand () % int(distributions.size ()));
    double p;
    double pSum;
    Distribution<double> weightedP;

    weightedP.fill (0.0, distributions.size ());

    // pick random 1st cluster
    clusters.emplace_back (distributions[distIdx]);
    clusterAssignments.emplace_back ();

    for (size_t k = 1; k < kMax; k++) {
        assignDistributions ();

        // calculate weighted probabillities
        for (size_t i = 0; i < clusterAssignments.size (); i++)
        for (size_t j = 0; j < clusterAssignments[i].size (); j++) {
            distIdx = clusterAssignments [i][j];
            weightedP[distIdx] = Distribution<double>::emd (
                distributions[distIdx],
                clusters[i]
            );
        }
        weightedP *= weightedP;
        weightedP /= weightedP.sum ();

        // select new cluster based on weighted probabillites
        p = double (rand ()) / double (RAND_MAX);
        pSum = 0.0;
        for (size_t i = 0; i < distributions.size (); i++) {
            pSum += weightedP[i];
            if (pSum >= p) {
                clusters.emplace_back (distributions[i]);
                clusterAssignments.emplace_back ();
                break;
            }
        }
    }
}

void PKMeans::clearClusterAssignments () {
    for (size_t i = 0; i < clusterAssignments.size (); i++) {
        clusterAssignments[i].clear ();
    }
}

size_t PKMeans::findClosestCluster (size_t distributionIdx) {
    size_t closestIdx = 0;
    double dist = Distribution<double>::emd (distributions[distributionIdx],
                                             clusters[0]);
    double newDist;
    for (size_t i = 1; i < clusters.size (); i++) {
        newDist = Distribution<double>::emd (distributions[distributionIdx],
                                             clusters[i]);
        if (newDist < dist) {
            closestIdx = i;
            dist = newDist;
        }
    }
    return closestIdx;
}

void PKMeans::assignDistributions () {
    converged = true;
    clearClusterAssignments ();
    size_t closestClusterIdx;
    for (size_t i = 0; i < distributions.size (); i++) {
        closestClusterIdx = findClosestCluster (i);
        clusterAssignments[closestClusterIdx].emplace_back (i);
        if (closestClusterIdx != prevClusterAssignments[i]) {
            converged = false;
            prevClusterAssignments[i] = closestClusterIdx;
        }
    }
}

void PKMeans::computeClusterMean (size_t idx) {
    clusters[idx].fill (0.0);
    for (size_t i = 0; i < clusterAssignments[idx].size (); i++) {
        clusters[idx] += distributions [clusterAssignments [idx][i]];
    }
    clusters[idx] /= double(clusterAssignments[idx].size ());
}

void PKMeans::computeNewClusters () {
    for (size_t i = 0; i < clusters.size (); i++) {
        computeClusterMean (i);
    }
}

double PKMeans::calcObjFn () {
    double sum = 0.0;
    double dist = 0.0;
    for (size_t i = 0; i < clusterAssignments.size (); i++)
    for (size_t j = 0; j < clusterAssignments[i].size (); j++) {
        dist = Distribution<double>::emd (
            clusters[i], distributions [clusterAssignments[i][j]]
        );
        sum += dist;
    }
    return sum;
}

