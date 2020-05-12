#include <gtest/gtest.h>
#include <pkmeans/pkmeans.h>
#include <pkmeans/distribution.h>
#include <iostream>
#include <string>
#include <array>
#include <fstream>

namespace pkmeans {
class PKMeansTests : public ::testing::Test {
protected:
    // You can remove any or all of the following functions if its body
    // is empty.

    PKMeansTests () {
        srand (time (NULL));

        // Initialize random distributions
        for (size_t j = 0; j < dummyBuckets.size (); j++)
        for (size_t i = 0; i < dummyBuckets[j].size (); i++) {
            dummyBuckets[j][i] = rand () % 100;
        }

        // Create test distribution file
        std::ofstream f;
        f.open ("test_distribution.txt");
        for (size_t i = 0; i < dummyBuckets.size (); i++) {
            for (size_t j = 0; j < 49; j++) {
                f << dummyBuckets[i][j] << ' ';
            }
            f << dummyBuckets[i][49] << '\n';
        }
        f.close ();

        pkmeans.readDistributions ("test_distribution.txt");
    }

    virtual ~PKMeansTests () {
        // You can do clean-up work that doesn't throw exceptions here.
        if ((std::remove ("test_distribution.txt")) != 0)
            printf ("Could not remove test_distribution.txt\n");
    }

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:
    virtual void SetUp () {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    virtual void TearDown () {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    PKMeans pkmeans;
    std::array<std::array<float, 50>, 4> dummyBuckets;
};

TEST_F (PKMeansTests, ReadDistributions) {
    ASSERT_EQ(4, pkmeans.distributions.size ());

    for (size_t i = 0; i < dummyBuckets.size (); i++)
    for (size_t j = 0; j < dummyBuckets[i].size (); j++) {
        EXPECT_FLOAT_EQ (dummyBuckets[i][j],
                          pkmeans.distributions[i][j])
            << "Distribution " << i << " different at bucket " << j << ".\n";
    }
}

TEST_F (PKMeansTests, InitClusters) {
    pkmeans.initClusters (2);

    ASSERT_EQ(2, pkmeans.clusters.size ());
    for (size_t i = 0; i < pkmeans.clusters.size (); i++) {
        ASSERT_EQ(dummyBuckets[0].size (), pkmeans.clusters[i].size ());
    }
}

TEST_F (PKMeansTests, InitNewClusters) {
    pkmeans.initClusters (2);
    pkmeans.initNewClusters ();

    ASSERT_EQ (2, pkmeans.newClusters.size ());
    EXPECT_EQ (pkmeans.clusters[0].size (),
               pkmeans.newClusters[0].size ());
    EXPECT_EQ (pkmeans.clusters[0].size (),
               pkmeans.newClusters[1].size ());
}

TEST_F (PKMeansTests, InitLowerBounds) {
    pkmeans.clusters.emplace_back ();
    pkmeans.clusters.emplace_back ();
    pkmeans.clusters.emplace_back ();
    pkmeans.clusters.emplace_back ();
    pkmeans.initLowerBounds ();

    ASSERT_EQ (pkmeans.distributions.size (),
               pkmeans.lowerBounds.size ());

    for (size_t x = 0; x < pkmeans.distributions.size (); x++) {
        EXPECT_EQ (pkmeans.clusters.size (),
                   pkmeans.lowerBounds[x].size ());
    }
}

TEST_F (PKMeansTests, InitClusterDists) {
    pkmeans.initClusters (2);
    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();

    ASSERT_EQ (pkmeans.clusters.size (),
               pkmeans.clusterDists.size ());
    for (size_t c = 0; c < pkmeans.clusterDists.size (); c++) {
        EXPECT_EQ (pkmeans.clusters.size (),
                   pkmeans.clusterDists[c].size ());
    }
}

TEST_F (PKMeansTests, InitSDists) {
    pkmeans.initClusters (2);
    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();

    ASSERT_EQ (pkmeans.clusters.size (), pkmeans.sDists.size ());
}

TEST_F (PKMeansTests, InitUpperBoundNeedsUpdate) {
    pkmeans.initClusters (2);
    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();

    ASSERT_EQ (pkmeans.distributions.size (), pkmeans.upperBoundNeedsUpdate.size ());
}

TEST_F (PKMeansTests, InitAssignments) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.clusters.emplace_back (pkmeans.distributions[0]);
    pkmeans.clusters.emplace_back (pkmeans.distributions[2]);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.computeDcDist (0, 0);
    pkmeans.computeDcDist (1, 0);
    pkmeans.computeDcDist (2, 0);
    pkmeans.computeDcDist (3, 0);
    pkmeans.computeDcDist (0, 1);
    pkmeans.computeDcDist (1, 1);
    pkmeans.computeDcDist (2, 1);
    pkmeans.computeDcDist (3, 1);

    pkmeans.initAssignments ();

    EXPECT_EQ(0, pkmeans.clusterMap[0]);
    EXPECT_EQ(0, pkmeans.clusterMap[1]);
    EXPECT_EQ(1, pkmeans.clusterMap[2]);
    EXPECT_EQ(1, pkmeans.clusterMap[3]);
}

TEST_F (PKMeansTests, InitUpperBounds) {
    pkmeans.initClusters (2);
    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initAssignments ();
    pkmeans.initUpperBounds ();

    EXPECT_EQ (pkmeans.distributions.size (),
               pkmeans.upperBounds.size ());
}

TEST_F (PKMeansTests, ComputeClusterDists) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;

    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initAssignments ();
    pkmeans.initUpperBounds ();
    pkmeans.computeClusterDists ();

    for (size_t c1 = 0; c1 < pkmeans.clusters.size (); c1++)
    for (size_t c2 = 0; c2 < pkmeans.clusters.size (); c2++) {
        EXPECT_EQ (
            Distribution<float>::emd (
                pkmeans.clusters[c1], pkmeans.clusters[c2]),
            pkmeans.clusterDists[c1][c2]);
    }
    EXPECT_EQ (
        0.5 * Distribution<float>::emd (
            pkmeans.clusters[0], pkmeans.clusters[1]),
        pkmeans.sDists[0]);
    EXPECT_EQ (
        0.5 * Distribution<float>::emd (
            pkmeans.clusters[0], pkmeans.clusters[1]),
        pkmeans.sDists[1]);
}

TEST_F (PKMeansTests, ResetUpperBoundNeedsUpdate) {
    pkmeans.initClusters (2);
    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initAssignments ();
    pkmeans.initUpperBounds ();
    pkmeans.upperBoundNeedsUpdate[0] = false;
    pkmeans.upperBoundNeedsUpdate[1] = false;

    pkmeans.resetUpperBoundNeedsUpdate ();

    for (size_t i = 0; i < pkmeans.upperBoundNeedsUpdate.size(); i++) {
        EXPECT_TRUE (pkmeans.upperBoundNeedsUpdate[i]);
    }
}

TEST_F (PKMeansTests, AssignNewClusters) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.initNewClusters ();
    pkmeans.newClusters[0].fill (0.0);
    pkmeans.newClusters[1].fill (0.0);
    pkmeans.newClusters[0][0] = 1.0;
    pkmeans.newClusters[0][49] = 1.0;
    pkmeans.newClusters[1][23] = 1.0;
    pkmeans.newClusters[1][22] = 1.0;

    pkmeans.assignNewClusters ();
    for (size_t c = 0; c < pkmeans.clusters.size (); c++) {
        EXPECT_EQ (pkmeans.newClusters[c], pkmeans.clusters[c]);
    }
}

TEST_F (PKMeansTests, ComputeLowerBounds) {
    pkmeans.initClusters (2);
    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initAssignments ();
    pkmeans.initUpperBounds ();
    pkmeans.computeNewClusters ();
    pkmeans.computeClusterDists ();
    pkmeans.computeLowerBounds ();

    for (size_t x = 0; x < pkmeans.distributions.size (); x++)
    for (size_t c = 0; c < pkmeans.clusters.size (); c++) {
        EXPECT_GE (
            Distribution<float>::emd (
                pkmeans.distributions[x],
                pkmeans.clusters[c]
            ),
            pkmeans.lowerBounds[x][c]
        );
        EXPECT_LE (0, pkmeans.lowerBounds[x][c]);
    }
}

TEST_F (PKMeansTests, ComputeUpperBounds) {
    pkmeans.initClusters (2);
    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initAssignments ();
    pkmeans.initUpperBounds ();
    pkmeans.computeNewClusters ();
    pkmeans.computeClusterDists ();
    pkmeans.computeUpperBounds ();

    for (size_t x = 0; x < pkmeans.distributions.size (); x++) {
        EXPECT_LE (
            Distribution<float>::emd (
                pkmeans.distributions[x],
                pkmeans.clusters[pkmeans.getCluster (x)]
            ),
            pkmeans.upperBounds[x]
        );
        EXPECT_LE (0, pkmeans.upperBounds[x]);
    }
}

TEST_F (PKMeansTests, NeedsClusterUpdate) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;

    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initAssignments ();
    pkmeans.initUpperBounds ();
    pkmeans.computeClusterDists ();
    pkmeans.assignDistributions ();

    pkmeans.computeNewClusters ();
    pkmeans.newClusters[0].fill (0.0);
    pkmeans.newClusters[1].fill (0.0);
    pkmeans.newClusters[1][0] = 1.0;
    pkmeans.newClusters[1][49] = 1.0;
    pkmeans.newClusters[0][23] = 1.0;
    pkmeans.newClusters[0][22] = 1.0;

    pkmeans.computeLowerBounds ();
    pkmeans.computeUpperBounds ();
    pkmeans.resetUpperBoundNeedsUpdate ();
    pkmeans.assignNewClusters ();
    pkmeans.computeClusterDists ();

    for (size_t x = 0; x < pkmeans.distributions.size (); x++)
    for (size_t c = 0; c < pkmeans.clusters.size (); c++) {
        if (c != pkmeans.getCluster (x)) {
            EXPECT_TRUE (pkmeans.needsClusterUpdateApprox (x, c));
            EXPECT_TRUE (pkmeans.needsClusterUpdate (x, c));
        }
        else {
            EXPECT_FALSE (pkmeans.needsClusterUpdate (x, c));
        }
    }
    pkmeans.assignDistributions ();
    pkmeans.computeNewClusters ();
    pkmeans.newClusters[0].fill (0.0);
    pkmeans.newClusters[1].fill (0.0);
    pkmeans.newClusters[1][0] = 1.0;
    pkmeans.newClusters[1][49] = 1.0;
    pkmeans.newClusters[0][23] = 1.0;
    pkmeans.newClusters[0][22] = 1.0;
    pkmeans.computeLowerBounds ();
    pkmeans.computeUpperBounds ();
    pkmeans.resetUpperBoundNeedsUpdate ();
    pkmeans.assignNewClusters ();
    pkmeans.computeClusterDists ();

    for (size_t x = 0; x < pkmeans.distributions.size (); x++)
    for (size_t c = 0; c < pkmeans.clusters.size (); c++) {
        EXPECT_FALSE (pkmeans.needsClusterUpdate (x, c));
    }
}

TEST_F (PKMeansTests, FindClosestInitCluster) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;

    pkmeans.initNewClusters ();
    pkmeans.initClusterDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initUpperBounds ();
    pkmeans.assignDistributions ();

    EXPECT_EQ (0, pkmeans.findClosestInitCluster (0));
    EXPECT_EQ (0, pkmeans.findClosestInitCluster (1));
    EXPECT_EQ (1, pkmeans.findClosestInitCluster (2));
    EXPECT_EQ (1, pkmeans.findClosestInitCluster (3));
}

TEST_F (PKMeansTests, GetCluster) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;

    pkmeans.initNewClusters ();
    pkmeans.initClusterDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initUpperBounds ();
    pkmeans.assignDistributions ();

    EXPECT_EQ (0, pkmeans.getCluster (0));
    EXPECT_EQ (0, pkmeans.getCluster (1));
    EXPECT_EQ (1, pkmeans.getCluster (2));
    EXPECT_EQ (1, pkmeans.getCluster (3));
}

TEST_F (PKMeansTests, ComputeDcDist) {
    pkmeans.initClusters (2);
    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initAssignments ();
    pkmeans.initUpperBounds ();

    for (size_t x = 0; x < pkmeans.distributions.size (); x++)
    for (size_t c = 0; c < pkmeans.clusters.size (); c++) {
        EXPECT_EQ (
            Distribution<float>::emd (
                pkmeans.distributions[x],
                pkmeans.clusters[c]
            ),
            pkmeans.computeDcDist (x, c)
        );
    }
}

TEST_F (PKMeansTests, CDist) {
    pkmeans.initClusters (2);
    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initAssignments ();
    pkmeans.initUpperBounds ();
    pkmeans.computeClusterDists ();

    for (size_t c1 = 0; c1 < pkmeans.clusters.size (); c1++)
    for (size_t c2 = 0; c2 < pkmeans.clusters.size (); c2++) {
        EXPECT_EQ (
            Distribution<float>::emd (
                pkmeans.clusters[c1],
                pkmeans.clusters[c2]
            ),
            pkmeans.cDist(c1, c2)
        );
    }
}

TEST_F (PKMeansTests, FindClosestCluster) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;

    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initAssignments ();
    pkmeans.initUpperBounds ();

    EXPECT_EQ (0, pkmeans.findClosestCluster (0))
        << "d(dist0,c0)="
        << Distribution<float>::emd (pkmeans.distributions[0],
                                      pkmeans.clusters[0])
        << "; d(dist0,c1)="
        << Distribution<float>::emd (pkmeans.distributions[0],
                                      pkmeans.clusters[1])
        << '\n';
    EXPECT_EQ (0, pkmeans.findClosestCluster (1))
        << "d(dist1,c0)="
        << Distribution<float>::emd (pkmeans.distributions[1],
                                      pkmeans.clusters[0])
        << "; d(dist1,c1)="
        << Distribution<float>::emd (pkmeans.distributions[1],
                                      pkmeans.clusters[1])
        << '\n';
    EXPECT_EQ (1, pkmeans.findClosestCluster (2))
        << "d(dist2,c0)="
        << Distribution<float>::emd (pkmeans.distributions[2],
                                      pkmeans.clusters[0])
        << "; d(dist2,c1)="
        << Distribution<float>::emd (pkmeans.distributions[2],
                                      pkmeans.clusters[1])
        << '\n';
    EXPECT_EQ (1, pkmeans.findClosestCluster (3))
        << "d(dist3,c0)="
        << Distribution<float>::emd (pkmeans.distributions[3],
                                      pkmeans.clusters[0])
        << "; d(dist3,c1)="
        << Distribution<float>::emd (pkmeans.distributions[3],
                                      pkmeans.clusters[1])
        << '\n';
}

TEST_F (PKMeansTests, AssignDistributions) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;

    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initAssignments ();
    pkmeans.initUpperBounds ();
    pkmeans.computeClusterDists ();

    pkmeans.assignDistributions ();

    ASSERT_EQ(2, pkmeans.clusterAssignments.size ());
    ASSERT_EQ(2, pkmeans.clusterAssignments[0].size ());
    ASSERT_EQ(2, pkmeans.clusterAssignments[1].size ());

    EXPECT_EQ(0, pkmeans.clusterAssignments[0][0]);
    EXPECT_EQ(1, pkmeans.clusterAssignments[0][1]);
    EXPECT_EQ(2, pkmeans.clusterAssignments[1][0]);
    EXPECT_EQ(3, pkmeans.clusterAssignments[1][1]);
}

TEST_F (PKMeansTests, ClearClusterAssignments) {
    pkmeans.initClusters (2);
    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initAssignments ();
    pkmeans.initUpperBounds ();
    pkmeans.computeClusterDists ();
    pkmeans.assignDistributions ();

    ASSERT_GT(pkmeans.clusterAssignments.size (), 0.0);
    for (size_t i = 0; i < pkmeans.clusterAssignments.size (); i++) {
        ASSERT_GT(pkmeans.clusterAssignments[i].size (), 0.0)
            << "cluster " << i << " has 0 assigned distributions.\n";
    }

    pkmeans.clearClusterAssignments ();

    ASSERT_GT(pkmeans.clusterAssignments.size (), 0.0);
    for (size_t i = 0; i < pkmeans.clusterAssignments.size (); i++) {
        ASSERT_EQ(pkmeans.clusterAssignments[i].size (), 0.0);
    }
}

TEST_F (PKMeansTests, ComputeClusterMean) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;

    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initAssignments ();
    pkmeans.initUpperBounds ();
    pkmeans.computeClusterDists ();
    pkmeans.assignDistributions ();

    Distribution<float> expected;
    Distribution<float> result;

    pkmeans.computeClusterMean (0);
    expected = (pkmeans.distributions[0] + pkmeans.distributions[1]) / 2.0;
    result = pkmeans.newClusters[0];
    EXPECT_EQ (expected, result);

    pkmeans.computeClusterMean (1);
    expected = (pkmeans.distributions[2] + pkmeans.distributions[3]) / 2.0;
    result = pkmeans.newClusters[1];
    EXPECT_EQ (expected, result);
}

TEST_F (PKMeansTests, ComputeNewClusters) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;

    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initAssignments ();
    pkmeans.initUpperBounds ();
    pkmeans.computeClusterDists ();
    pkmeans.assignDistributions ();
    pkmeans.computeNewClusters ();

    Distribution<float> expected;
    Distribution<float> result;

    expected = (pkmeans.distributions[0] + pkmeans.distributions[1]) / 2.0;
    result = pkmeans.newClusters[0];
    EXPECT_EQ (expected, result);

    expected = (pkmeans.distributions[2] + pkmeans.distributions[3]) / 2.0;
    result = pkmeans.newClusters[1];
    EXPECT_EQ (expected, result);
}

TEST_F (PKMeansTests, CalcObjFn) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;

    pkmeans.initNewClusters ();
    pkmeans.initLowerBounds ();
    pkmeans.initClusterDists ();
    pkmeans.initSDists ();
    pkmeans.initUpperBoundNeedsUpdate ();
    pkmeans.initAssignments ();
    pkmeans.initUpperBounds ();
    pkmeans.computeClusterDists ();
    pkmeans.assignDistributions ();

    EXPECT_FLOAT_EQ (15.0, pkmeans.calcObjFn ());

    pkmeans.computeNewClusters ();
    pkmeans.newClusters[0].fill (0.0);
    pkmeans.newClusters[1].fill (0.0);
    pkmeans.newClusters[0][2] = 1.0;
    pkmeans.newClusters[0][47] = 1.0;
    pkmeans.newClusters[1][24] = 1.0;
    pkmeans.newClusters[1][21] = 1.0;

    pkmeans.computeLowerBounds ();
    pkmeans.computeUpperBounds ();
    pkmeans.resetUpperBoundNeedsUpdate ();
    pkmeans.assignNewClusters ();

    EXPECT_FLOAT_EQ (17.0, pkmeans.calcObjFn ());
}

}
