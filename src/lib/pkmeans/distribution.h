#ifndef __DISTRIBUTION_H_INCLUDED_
#define __DISTRIBUTION_H_INCLUDED_

#include <tgmath.h>

#include <algorithm>
#include <boost/functional/hash.hpp>
#include <sstream>
#include <vector>

namespace pkmeans {
template <class T>
struct Distribution {
 private:
  std::vector<T> buckets;

 public:
  static bool euclidean;

  Distribution(){};

  void emplace_back(const T& val) { buckets.emplace_back(val); }

  void clear() { buckets.clear(); }

  size_t size() const { return buckets.size(); };

  size_t hash() const {
    return boost::hash_range(buckets.begin(), buckets.end());
  }

  void fill(T value) { std::fill(buckets.begin(), buckets.end(), value); };

  void fill(T value, size_t N) {
    buckets.clear();
    for (size_t i = 0; i < N; i++) buckets.emplace_back(value);
  };

  T sum() const {
    T retSum = 0;
    for (size_t i = 0; i < size(); i++) {
      retSum += buckets[i];
    }
    return retSum;
  }

  T absSum() const {
    T retSum = 0;
    for (size_t i = 0; i < size(); i++) {
      retSum += fabs(buckets[i]);
    }
    return retSum;
  }

  T& operator[](size_t idx) { return buckets[idx]; };

  const T& operator[](size_t idx) const { return buckets[idx]; };

  Distribution& operator=(const Distribution& other) {
    if (this != &other) {
      buckets = other.buckets;
    }
    return *this;
  }

  Distribution& operator+=(const Distribution& other) {
    for (size_t i = 0; i < size(); i++) {
      buckets[i] += other.buckets[i];
    }
    return *this;
  }

  Distribution& operator-=(const Distribution& other) {
    for (size_t i = 0; i < size(); i++) {
      buckets[i] -= other.buckets[i];
    }
    return *this;
  };

  Distribution& operator*=(const Distribution& other) {
    for (size_t i = 0; i < size(); i++) {
      buckets[i] *= other.buckets[i];
    }
    return *this;
  };

  Distribution& operator*=(const T& rhs) {
    for (size_t i = 0; i < size(); i++) {
      buckets[i] *= rhs;
    }
    return *this;
  };

  Distribution& operator/=(const Distribution& other) {
    for (size_t i = 0; i < size(); i++) {
      buckets[i] /= other.buckets[i];
    }
    return *this;
  };

  Distribution& operator/=(const T& rhs) {
    for (size_t i = 0; i < size(); i++) {
      buckets[i] /= rhs;
    }
    return *this;
  };

  friend bool operator==(const Distribution& lhs, const Distribution& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); i++) {
      if (lhs[i] != rhs[i]) return false;
    }
    return true;
  }

  friend bool operator!=(const Distribution& lhs, const Distribution& rhs) {
    return !(lhs == rhs);
  }

  friend std::istream& operator>>(std::istream& is, Distribution& obj) {
    T tempVal;
    std::string buf;
    std::getline(is, buf);
    std::istringstream iss(buf);
    while (iss) {
      if (!(iss >> tempVal)) break;
      obj.buckets.push_back(tempVal);
    }
    return is;
  };

  friend std::ostream& operator<<(std::ostream& os, const Distribution& obj) {
    for (size_t i = 0; i < obj.buckets.size() - 1; i++) {
      os << obj.buckets[i] << ' ';
    }
    os << obj.buckets[obj.buckets.size() - 1];
    return os;
  };

  friend Distribution operator+(Distribution lhs, const Distribution& rhs) {
    lhs += rhs;
    return lhs;
  };

  friend Distribution operator-(Distribution lhs, const Distribution& rhs) {
    lhs -= rhs;
    return lhs;
  };

  friend Distribution operator*(Distribution lhs, const Distribution& rhs) {
    lhs *= rhs;
    return lhs;
  };

  friend Distribution operator*(Distribution lhs, const T& rhs) {
    for (size_t i = 0; i < lhs.size(); i++) {
      lhs.buckets[i] *= rhs;
    }
    return lhs;
  };

  friend Distribution operator*(const T& lhs, Distribution rhs) {
    for (size_t i = 0; i < rhs.size(); i++) {
      rhs.buckets[i] *= lhs;
    }
    return rhs;
  };

  friend Distribution operator/(Distribution lhs, const Distribution& rhs) {
    lhs /= rhs;
    return lhs;
  };

  friend Distribution operator/(Distribution lhs, const T& rhs) {
    for (size_t i = 0; i < lhs.size(); i++) {
      lhs.buckets[i] /= rhs;
    }
    return lhs;
  };

  friend Distribution operator/(const T& lhs, Distribution rhs) {
    for (size_t i = 0; i < rhs.size(); i++) {
      rhs.buckets[i] /= lhs;
    }
    return rhs;
  };

  static float emd(const Distribution& d1, const Distribution& d2) {
    float sum = 0.0;
    if (Distribution<T>::euclidean) {
      for (size_t i = 0; i < d1.size(); i++) {
        sum +=
            (d1.buckets[i] - d2.buckets[i]) * (d1.buckets[i] - d2.buckets[i]);
      }
      sum = sqrt(sum);
    } else {
      float emd_i = 0.0;
      for (size_t i = 0; i < d1.size(); i++) {
        emd_i += d1.buckets[i] - d2.buckets[i];
        sum += fabs(emd_i);
      }
    }
    return sum;
  };
};
template <class T>
bool Distribution<T>::euclidean = false;
}  // namespace pkmeans

#endif  // __DISTRIBUTION_H_INCLUDED_
