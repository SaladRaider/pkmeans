#include <gtest/gtest.h>
#include <pkmeans/distribution.h>
#include <string>
#include <array>
#include <time.h>
#include <fstream>

using namespace pkmeans;

class DistributionTests : public ::testing::Test {
protected:
    // You can remove any or all of the following functions if its body
    // is empty.

    DistributionTests () {
        srand (time (NULL));

        // You can do set-up work for each test here.
        // Initialize random distributions
        for (size_t j = 0; j < 4; j++)
        for (size_t i = 0; i < 50; i++) {
            dummyBuckets[j][i] = rand () % 100;
        }

        // Create test distribution file
        std::ofstream f;
        f.open("test_distribution.txt");
        for (size_t j = 0; j < 4; j++)
        for (size_t i = 0; i < dummyBuckets.size (); i++) {
            for (size_t j = 0; j < 49; j++) {
                f << dummyBuckets[i][j] << ' ';
            }
            f << dummyBuckets[i][49] << '\n';
        }
        f.close ();

        // Stream test distribution file
        std::ifstream infile;
        std::string fileBuffer;
        infile.open ("test_distribution.txt");
        for (size_t j = 0; j < 4; j++) {
            infile >> distributions[j];
        }
        infile.close ();
    }

    virtual ~DistributionTests () {
        // You can do clean-up work that doesn't throw exceptions here.
        std::remove ("test_dummyBuckets.txt");
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

    static constexpr size_t numDistributions = 4;
    std::array<Distribution<double>, numDistributions> distributions;
    std::array<std::array<double, 50>, numDistributions> dummyBuckets;
};

TEST_F (DistributionTests, SizeTest) {
    for (size_t i = 0; i < distributions.size (); i++) {
        EXPECT_EQ (50, distributions[i].size ());
    }
}

TEST_F (DistributionTests, FillTest) {
    for (size_t i = 0; i < distributions.size (); i++) {
        double randVal = double(rand () % 100);
        distributions[i].fill (randVal);
        for (size_t j = 0; j < distributions[i].size (); j++) {
            EXPECT_DOUBLE_EQ (randVal, distributions[i][j]);
        }
    }
}

TEST_F (DistributionTests, ArithmeticTest) {
    Distribution<double> sum;
    sum = distributions[0] + distributions[1]
        + distributions[2] + distributions[3];
    ASSERT_EQ (distributions[0].size (), sum.size ());
    for (size_t j = 0; j < distributions[0].size (); j++) {
        double expectedVal = distributions[0][j] + distributions[1][j]
            + distributions[2][j] + distributions[3][j];
        EXPECT_DOUBLE_EQ (expectedVal, sum[j]);
    }

    sum.fill (0.0);
    sum += distributions[0];
    sum += distributions[1];
    sum += distributions[2];
    sum += distributions[3];
    ASSERT_EQ (distributions[0].size (), sum.size ());
    for (size_t j = 0; j < distributions[0].size (); j++) {
        double expectedVal = distributions[0][j] + distributions[1][j]
            + distributions[2][j] + distributions[3][j];
        EXPECT_DOUBLE_EQ (expectedVal, sum[j]);
    }

    sum = distributions[0] - distributions[1] - distributions[2]
        - distributions[3];
    ASSERT_EQ (distributions[0].size (), sum.size ());
    for (size_t j = 0; j < distributions[0].size (); j++) {
        double expectedVal = distributions[0][j] - distributions[1][j]
            - distributions[2][j] - distributions[3][j];
        EXPECT_DOUBLE_EQ (expectedVal, sum[j]);
    }

    sum.fill (0.0);
    sum -= distributions[0];
    sum -= distributions[1];
    sum -= distributions[2];
    sum -= distributions[3];
    ASSERT_EQ (distributions[0].size (), sum.size ());
    for (size_t j = 0; j < distributions[0].size (); j++) {
        double expectedVal = 0;
        expectedVal = -distributions[0][j] - distributions[1][j]
            - distributions[2][j] - distributions[3][j];
        EXPECT_DOUBLE_EQ (expectedVal, sum[j]) << "sum[" << j << "]\n";
    }

    sum = distributions[0] * distributions[1] * distributions[2]
        * distributions[3];
    ASSERT_EQ (distributions[0].size (), sum.size ());
    for (size_t j = 0; j < distributions[0].size (); j++) {
        double expectedVal = distributions[0][j] * distributions[1][j]
            * distributions[2][j] * distributions[3][j];
        EXPECT_DOUBLE_EQ (expectedVal, sum[j]);
    }

    sum.fill (1.0);
    sum *= distributions[0];
    sum *= distributions[1];
    sum *= distributions[2];
    sum *= distributions[3];
    ASSERT_EQ (distributions[0].size (), sum.size ());
    for (size_t j = 0; j < distributions[0].size (); j++) {
        double expectedVal = 0;
        expectedVal = distributions[0][j] * distributions[1][j]
            * distributions[2][j] * distributions[3][j];
        EXPECT_DOUBLE_EQ (expectedVal, sum[j]) << "sum[" << j << "]\n";
    }

    sum = distributions[0] * 1.0 * distributions[1] * 2.0
        * distributions[2] * 3.0 * distributions[3] * 4.0;
    ASSERT_EQ (distributions[0].size (), sum.size ());
    for (size_t j = 0; j < distributions[0].size (); j++) {
        double expectedVal = distributions[0][j] * distributions[1][j]
            * distributions[2][j] * distributions[3][j] * 24.0;
        EXPECT_DOUBLE_EQ (expectedVal, sum[j]);
    }

    sum.fill (1.0);
    sum *= 1.0;
    sum *= 2.0;
    sum *= 3.0;
    sum *= 4.0;
    ASSERT_EQ (distributions[0].size (), sum.size ());
    for (size_t j = 0; j < distributions[0].size (); j++) {
        EXPECT_DOUBLE_EQ (24.0, sum[j]) << "sum[" << j << "]\n";
    }

    sum = distributions[0] / distributions[1] / distributions[2]
        / distributions[3];
    ASSERT_EQ (distributions[0].size (), sum.size ());
    for (size_t j = 0; j < distributions[0].size (); j++) {
        double expectedVal = distributions[0][j] / distributions[1][j]
            / distributions[2][j] / distributions[3][j];
        EXPECT_DOUBLE_EQ (expectedVal, sum[j]);
    }

    sum.fill (1.0);
    sum /= distributions[0];
    sum /= distributions[1];
    sum /= distributions[2];
    sum /= distributions[3];
    ASSERT_EQ (distributions[0].size (), sum.size ());
    for (size_t j = 0; j < distributions[0].size (); j++) {
        double expectedVal = 0;
        expectedVal = 1.0 / distributions[0][j] / distributions[1][j]
            / distributions[2][j] / distributions[3][j];
        EXPECT_DOUBLE_EQ (expectedVal, sum[j]) << "sum[" << j << "]\n";
    }

    sum = distributions[0] / 1.0 / distributions[1] / 2.0
        / distributions[2] / 3.0 / distributions[3] / 4.0;
    ASSERT_EQ (distributions[0].size (), sum.size ());
    for (size_t j = 0; j < distributions[0].size (); j++) {
        double expectedVal = distributions[0][j] / distributions[1][j]
            / distributions[2][j] / distributions[3][j] / 24.0;
        EXPECT_DOUBLE_EQ (expectedVal, sum[j]);
    }

    sum.fill (1.0);
    sum /= 1.0;
    sum /= 2.0;
    sum /= 3.0;
    sum /= 4.0;
    ASSERT_EQ (distributions[0].size (), sum.size ());
    for (size_t j = 0; j < distributions[0].size (); j++) {
        EXPECT_DOUBLE_EQ (1.0/24.0, sum[j]) << "sum[" << j << "]\n";
    }
}
