#ifndef __DISTRIBUTION_H_INCLUDED_
#define __DISTRIBUTION_H_INCLUDED_

#include <vector>
//#include <thrust/device_vector.h>
//#include <thrust/sequence.h>
//#include <thrust/execution_policy.h>
#include <sstream>
#include <algorithm>
#include <tgmath.h>

namespace pkmeans {

template<typename T>
struct absdiff_functor {
    typedef T first_argument_type;
    typedef T second_argument_type;
    typedef T result_type;
    __host__ __device__
    T operator () (const T &x, const T &y) const {
        return fabs (x - y);
    }
};

template <class T>
struct Distribution {
    private:
        std::vector<T> buckets;
    public:
        Distribution () {};

        void emplace_back (const T &val) {
            buckets.emplace_back (val);
        }

        void clear () {
            buckets.clear ();
        }

        size_t size () const {
            return buckets.size ();
        };

        void fill (T value) {
            std::fill (buckets.begin (), buckets.end (), value);
        };

        void fill (T value, size_t N) {
            buckets.clear ();
            for (size_t i = 0; i < N; i++)
                buckets.emplace_back (value);
        };

        T sum () const {
            T retSum = 0;
            for (size_t i = 0; i < size (); i++) {
                retSum += buckets[i];
            }
            return retSum;
        }

        T absSum () const {
            T retSum = 0;
            for (size_t i = 0; i < size (); i++) {
                retSum += fabs (buckets[i]);
            }
            return retSum;
        }

        T& operator[] (size_t idx) {
            return buckets[idx];
        };

        const T& operator[] (size_t idx) const {
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

        friend bool operator== (const Distribution &lhs, const Distribution &rhs) {
            if (lhs.size () != rhs.size ())
                return false;
            for (size_t i = 0; i < lhs.size (); i++) {
                if (lhs[i] != rhs[i])
                    return false;
            }
            return true;
        }

        friend bool operator!= (const Distribution &lhs, const Distribution &rhs) {
            return !(lhs == rhs);
        }

        friend std::istream& operator>> (std::istream &is, Distribution &obj) {
            T tempVal;
            std::string buf;
            std::getline (is, buf);
            std::istringstream iss (buf);
            while (iss) {
                if (!(iss >> tempVal))
                    break;
                obj.buckets.push_back (tempVal);
            }
            return is;
        };

        friend std::ostream& operator<< (std::ostream &os, const Distribution &obj) {
            for (size_t i = 0; i < obj.buckets.size () - 1; i++) {
                os << obj.buckets[i] << ' ';
            }
            os << obj.buckets[obj.buckets.size () - 1];
            return os;
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

        static T emd (const Distribution &d1, const Distribution &d2) {
            /*thrust::device_vector<T> buckets1 (d1.buckets);
            thrust::device_vector<T> buckets2 (d2.buckets);
            thrust::device_vector<T> scalars (buckets1.size ());
            thrust::sequence (thrust::device, scalars.begin (), scalars.end (), 1);
            thrust::transform (
                buckets1.begin (), buckets1.end (),
                buckets2.begin (), buckets2.begin (),
                absdiff_functor<T> {}
            );
            thrust::transform (
                buckets1.begin (), buckets1.end (),
                buckets2.begin (), buckets2.begin (),
                thrust::multiplies<T> ()
            );
            T sum = thrust::reduce (buckets2.begin (), buckets2.end (), T (0), thrust::plus<T> ());
            */
            T sum = 0;
            return sum;
        };
};
}

#endif // __DISTRIBUTION_H_INCLUDED_
