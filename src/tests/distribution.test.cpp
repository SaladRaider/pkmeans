#include <gtest/gtest.h>
#include <pkmeans/distribution.h>
#include <time.h>

#include <array>
#include <fstream>
#include <string>

using namespace pkmeans;

class DistributionTests : public ::testing::Test {
 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  DistributionTests() {
    srand(time(NULL));

    // Initialize random distributions
    for (size_t j = 0; j < 4; j++)
      for (size_t i = 0; i < 50; i++) {
        dummyBuckets[j][i] = rand() % 100;
      }

    // Create test distribution file
    std::ofstream f;
    f.open("test_distribution.txt");
    for (size_t j = 0; j < 4; j++)
      for (size_t i = 0; i < dummyBuckets.size(); i++) {
        for (size_t j = 0; j < 49; j++) {
          f << dummyBuckets[i][j] << ' ';
        }
        f << dummyBuckets[i][49] << '\n';
      }
    f.close();

    // Stream test distribution file
    std::ifstream infile;
    std::string fileBuffer;
    infile.open("test_distribution.txt");
    for (size_t j = 0; j < 4; j++) {
      infile >> distributions[j];
    }
    infile.close();
  }

  virtual ~DistributionTests() {
    // You can do clean-up work that doesn't throw exceptions here.
    if ((std::remove("test_distribution.txt")) != 0)
      printf("Could not remove test_distribution.txt\n");
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:
  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  static constexpr size_t numDistributions = 4;
  std::array<Distribution<double>, numDistributions> distributions;
  std::array<std::array<double, 50>, numDistributions> dummyBuckets;
};

TEST_F(DistributionTests, SizeTest) {
  for (size_t i = 0; i < distributions.size(); i++) {
    EXPECT_EQ(50, distributions[i].size());
  }
}

TEST_F(DistributionTests, Fill) {
  for (size_t i = 0; i < distributions.size(); i++) {
    double randVal = double(rand() % 100);
    distributions[i].fill(randVal);
    for (size_t j = 0; j < distributions[i].size(); j++) {
      EXPECT_DOUBLE_EQ(randVal, distributions[i][j]);
    }
  }
}

TEST_F(DistributionTests, FillN) {
  for (size_t i = 0; i < distributions.size(); i++) {
    double randVal = double(rand() % 100);
    distributions[i].fill(randVal, 50);
    EXPECT_DOUBLE_EQ(50, distributions[i].size());
    for (size_t j = 0; j < distributions[i].size(); j++) {
      EXPECT_DOUBLE_EQ(randVal, distributions[i][j]);
    }

    distributions[i].fill(randVal, 100);
    EXPECT_DOUBLE_EQ(100, distributions[i].size());
    for (size_t j = 0; j < distributions[i].size(); j++) {
      EXPECT_DOUBLE_EQ(randVal, distributions[i][j]);
    }

    distributions[i].fill(randVal, 5);
    EXPECT_DOUBLE_EQ(5, distributions[i].size());
    for (size_t j = 0; j < distributions[i].size(); j++) {
      EXPECT_DOUBLE_EQ(randVal, distributions[i][j]);
    }
  }
}

TEST_F(DistributionTests, Addition) {
  Distribution<double> sum;
  sum =
      distributions[0] + distributions[1] + distributions[2] + distributions[3];
  ASSERT_EQ(distributions[0].size(), sum.size());
  for (size_t j = 0; j < distributions[0].size(); j++) {
    double expectedVal = distributions[0][j] + distributions[1][j] +
                         distributions[2][j] + distributions[3][j];
    EXPECT_DOUBLE_EQ(expectedVal, sum[j]);
  }

  sum.fill(0.0);
  sum += distributions[0];
  sum += distributions[1];
  sum += distributions[2];
  sum += distributions[3];
  ASSERT_EQ(distributions[0].size(), sum.size());
  for (size_t j = 0; j < distributions[0].size(); j++) {
    double expectedVal = distributions[0][j] + distributions[1][j] +
                         distributions[2][j] + distributions[3][j];
    EXPECT_DOUBLE_EQ(expectedVal, sum[j]);
  }
}

TEST_F(DistributionTests, Subtraction) {
  Distribution<double> sum;
  sum =
      distributions[0] - distributions[1] - distributions[2] - distributions[3];
  ASSERT_EQ(distributions[0].size(), sum.size());
  for (size_t j = 0; j < distributions[0].size(); j++) {
    double expectedVal = distributions[0][j] - distributions[1][j] -
                         distributions[2][j] - distributions[3][j];
    EXPECT_DOUBLE_EQ(expectedVal, sum[j]);
  }

  sum.fill(0.0);
  sum -= distributions[0];
  sum -= distributions[1];
  sum -= distributions[2];
  sum -= distributions[3];
  ASSERT_EQ(distributions[0].size(), sum.size());
  for (size_t j = 0; j < distributions[0].size(); j++) {
    double expectedVal = 0;
    expectedVal = -distributions[0][j] - distributions[1][j] -
                  distributions[2][j] - distributions[3][j];
    EXPECT_DOUBLE_EQ(expectedVal, sum[j]) << "sum[" << j << "]\n";
  }
}

TEST_F(DistributionTests, Multiplication) {
  Distribution<double> sum;
  sum =
      distributions[0] * distributions[1] * distributions[2] * distributions[3];
  ASSERT_EQ(distributions[0].size(), sum.size());
  for (size_t j = 0; j < distributions[0].size(); j++) {
    double expectedVal = distributions[0][j] * distributions[1][j] *
                         distributions[2][j] * distributions[3][j];
    EXPECT_DOUBLE_EQ(expectedVal, sum[j]);
  }

  sum.fill(1.0);
  sum *= distributions[0];
  sum *= distributions[1];
  sum *= distributions[2];
  sum *= distributions[3];
  ASSERT_EQ(distributions[0].size(), sum.size());
  for (size_t j = 0; j < distributions[0].size(); j++) {
    double expectedVal = 0;
    expectedVal = distributions[0][j] * distributions[1][j] *
                  distributions[2][j] * distributions[3][j];
    EXPECT_DOUBLE_EQ(expectedVal, sum[j]) << "sum[" << j << "]\n";
  }

  sum = distributions[0] * 1.0 * distributions[1] * 2.0 * distributions[2] *
        3.0 * distributions[3] * 4.0;
  ASSERT_EQ(distributions[0].size(), sum.size());
  for (size_t j = 0; j < distributions[0].size(); j++) {
    double expectedVal = distributions[0][j] * distributions[1][j] *
                         distributions[2][j] * distributions[3][j] * 24.0;
    EXPECT_DOUBLE_EQ(expectedVal, sum[j]);
  }

  sum.fill(1.0);
  sum *= 1.0;
  sum *= 2.0;
  sum *= 3.0;
  sum *= 4.0;
  ASSERT_EQ(distributions[0].size(), sum.size());
  for (size_t j = 0; j < distributions[0].size(); j++) {
    EXPECT_DOUBLE_EQ(24.0, sum[j]) << "sum[" << j << "]\n";
  }
}

TEST_F(DistributionTests, Division) {
  Distribution<double> sum;
  sum =
      distributions[0] / distributions[1] / distributions[2] / distributions[3];
  ASSERT_EQ(distributions[0].size(), sum.size());
  for (size_t j = 0; j < distributions[0].size(); j++) {
    double expectedVal = distributions[0][j] / distributions[1][j] /
                         distributions[2][j] / distributions[3][j];
    EXPECT_DOUBLE_EQ(expectedVal, sum[j]);
  }

  sum.fill(1.0);
  sum /= distributions[0];
  sum /= distributions[1];
  sum /= distributions[2];
  sum /= distributions[3];
  ASSERT_EQ(distributions[0].size(), sum.size());
  for (size_t j = 0; j < distributions[0].size(); j++) {
    double expectedVal = 0;
    expectedVal = 1.0 / distributions[0][j] / distributions[1][j] /
                  distributions[2][j] / distributions[3][j];
    EXPECT_DOUBLE_EQ(expectedVal, sum[j]) << "sum[" << j << "]\n";
  }

  sum = distributions[0] / 1.0 / distributions[1] / 2.0 / distributions[2] /
        3.0 / distributions[3] / 4.0;
  ASSERT_EQ(distributions[0].size(), sum.size());
  for (size_t j = 0; j < distributions[0].size(); j++) {
    double expectedVal = distributions[0][j] / distributions[1][j] /
                         distributions[2][j] / distributions[3][j] / 24.0;
    EXPECT_DOUBLE_EQ(expectedVal, sum[j]);
  }

  sum.fill(1.0);
  sum /= 1.0;
  sum /= 2.0;
  sum /= 3.0;
  sum /= 4.0;
  ASSERT_EQ(distributions[0].size(), sum.size());
  for (size_t j = 0; j < distributions[0].size(); j++) {
    EXPECT_DOUBLE_EQ(1.0 / 24.0, sum[j]) << "sum[" << j << "]\n";
  }
}

TEST_F(DistributionTests, EMD) {
  std::array<std::array<double, numDistributions>, numDistributions> distMatrix;
  for (size_t i = 0; i < distMatrix.size(); i++)
    for (size_t j = 0; j < distMatrix[i].size(); j++) {
      distMatrix[i][j] =
          Distribution<double>::emd(distributions[i], distributions[j]);
    }

  // test if diagonal entries are 0
  for (size_t i = 0; i < distMatrix.size(); i++) {
    ASSERT_DOUBLE_EQ(0.0, distMatrix[i][i])
        << '(' << i << ',' << i << ") is non-zero.\n";
  }

  // test if matrix is symmetric and entries are non-negative
  for (size_t i = 0; i < distMatrix.size(); i++)
    for (size_t j = i; j < distMatrix[i].size(); j++) {
      ASSERT_GE(distMatrix[i][j], 0.0)
          << '(' << i << ',' << j << ") is negative.\n";
      ASSERT_DOUBLE_EQ(distMatrix[j][i], distMatrix[i][j])
          << '(' << i << ',' << j << ") is not symmetric.\n";
    }

  // test if triangle inequality is satisified
  double a, b, c;
  for (size_t i = 0; i < distMatrix.size(); i++)
    for (size_t j = 0; j < distMatrix.size(); j++)
      if (j != i)
        for (size_t k = j + 1; k < distMatrix.size(); k++)
          if (k != i) {
            a = distMatrix[i][j];
            b = distMatrix[i][k];
            c = distMatrix[j][k];
            ASSERT_GE(a + b, c) << '(' << i << ',' << j << ',' << k
                                << ") (a:" << a << ",b:" << b << ",c:" << c
                                << ") triangle inequality not satisfied.\n";
            ASSERT_GE(a + c, b) << '(' << i << ',' << j << ',' << k
                                << ") (a:" << a << ",b:" << b << ",c:" << c
                                << ") triangle inequality not satisfied.\n";
            ASSERT_GE(b + c, a) << '(' << i << ',' << j << ',' << k
                                << ") (a:" << a << ",b:" << b << ",c:" << c
                                << ") triangle inequality not satisfied.\n";
          }
}
