#ifndef ANN_TESTPROGRESS_HPP
#define ANN_TESTPROGRESS_HPP

#include <functional>
#include <sys/types.h>

//==============================================================================//

namespace ANN
{
  template <typename T>
  struct TestProgress {
      ulong currentSample;
      ulong totalSamples;
  };

  template <typename T>
  using TestCallback = std::function<void(const TestProgress<T>&)>;
}

//==============================================================================//

#endif // ANN_TESTPROGRESS_HPP
