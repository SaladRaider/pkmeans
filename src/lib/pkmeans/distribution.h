#ifndef __DISTRIBUTION_H_INCLUDED_
#define __DISTRIBUTION_H_INCLUDED_

#include <vector>
#include <sstream>
#include <algorithm>

namespace pkmeans {
template <class T>
struct Distribution {
    private:
        std::vector<T> buckets;
    public:
        Distribution () {};

        size_t size () const {
            return buckets.size ();
        };

        void fill (T value) {
            std::fill (buckets.begin(), buckets.end(), value);
        };

        T& operator[] (size_t idx) {
            return buckets[idx];
        };

        Distribution& operator= (const Distribution &other) {
            if (this != &other) {
                buckets = other.buckets;
            }
            return *this;
        }

        Distribution& operator+= (const Distribution &other) {
            for (size_t i = 0; i < size (); i++) {
                buckets[i] += other.buckets[i];
            }
            return *this;
        }

        Distribution& operator-= (const Distribution &other) {
            for (size_t i = 0; i < size (); i++) {
                buckets[i] -= other.buckets[i];
            }
            return *this;
        };

        Distribution& operator*= (const Distribution &other) {
            for (size_t i = 0; i < size (); i++) {
                buckets[i] *= other.buckets[i];
            }
            return *this;
        };

        Distribution& operator*= (const T &rhs) {
            for (size_t i = 0; i < size (); i++) {
                buckets[i] *= rhs;
            }
            return *this;
        };

        Distribution& operator/= (const Distribution &other) {
            for (size_t i = 0; i < size (); i++) {
                buckets[i] /= other.buckets[i];
            }
            return *this;
        };

        Distribution& operator/= (const T &rhs) {
            for (size_t i = 0; i < size (); i++) {
                buckets[i] /= rhs;
            }
            return *this;
        };

        friend std::istream& operator>> (std::istream &is, Distribution &obj) {
            T tempVal;
            std::string buf;
            std::getline (is, buf);
            std::istringstream iss (buf);
            while (iss) {
                if (!(iss >> tempVal))
                    break;
                obj.buckets.push_back(tempVal);
            }
            return is;
        };

        friend Distribution operator+ (Distribution lhs, const Distribution &rhs) {
            lhs += rhs;
            return lhs;
        };

        friend Distribution operator- (Distribution lhs, const Distribution &rhs) {
            lhs -= rhs;
            return lhs;
        };

        friend Distribution operator* (Distribution lhs, const Distribution &rhs) {
            lhs *= rhs;
            return lhs;
        };

        friend Distribution operator* (Distribution lhs, const T &rhs) {
            for (size_t i = 0; i < lhs.size (); i++) {
                lhs.buckets[i] *= rhs;
            }
            return lhs;
        };

        friend Distribution operator* (const T &lhs, Distribution rhs) {
            for (size_t i = 0; i < rhs.size (); i++) {
                rhs.buckets[i] *= lhs;
            }
            return rhs;
        };

        friend Distribution operator/ (Distribution lhs, const Distribution &rhs) {
            lhs /= rhs;
            return lhs;
        };

        friend Distribution operator/ (Distribution lhs, const T &rhs) {
            for (size_t i = 0; i < lhs.size (); i++) {
                lhs.buckets[i] /= rhs;
            }
            return lhs;
        };

        friend Distribution operator/ (const T &lhs, Distribution rhs) {
            for (size_t i = 0; i < rhs.size (); i++) {
                rhs.buckets[i] /= lhs;
            }
            return rhs;
        };

        static unsigned int emd (const Distribution &d1, const Distribution &d2) {
            return 0;
        };
};
}

#endif // __DISTRIBUTION_H_INCLUDED_
