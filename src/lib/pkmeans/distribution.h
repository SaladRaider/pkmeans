#ifndef __DISTRIBUTION_H_INCLUDED_
#define __DISTRIBUTION_H_INCLUDED_

#include <vector>

namespace pkmeans {
template <class T>
struct Distribution {
    private:
        std::vector<T> buckets;
    public:
        Distribution ();
        size_t size () const;
        void fill (T value);

        T& operator[] (size_t idx);
        Distribution& operator= (const Distribution &other);
        Distribution& operator+= (const Distribution &other);
        Distribution& operator-= (const Distribution &other);
        friend Distribution operator+ (Distribution lhs, const Distribution &rhs);
        friend Distribution operator- (Distribution lhs, const Distribution &rhs);

        static unsigned int emd (const Distribution &d1, const Distribution &d2);
};
}

#endif // __DISTRIBUTION_H_INCLUDED_
