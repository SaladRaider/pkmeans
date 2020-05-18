#ifndef __PROGRESSBAR_INCLUDED_
#define __PROGRESSBAR_INCLUDED_

#include <stddef.h>
#include <chrono>

namespace pkmeans {
template <size_t N>
class ProgressBar {
 private:
  static constexpr auto progressSize = size_t{N};
  static constexpr auto timeDecay = 0.8f;
  size_t progress;
  size_t prevProgress;
  float numShows;
  float elapsedSum;
  char progressBar[N + 1];
  std::chrono::steady_clock::time_point lastShownTime;

 public:
  ProgressBar();
  void show(float progressPercent);
};
}  // namespace pkmeans

#endif  // __PROGRESSBAR_INCLUDED_
