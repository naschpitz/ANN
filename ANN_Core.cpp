#include "ANN_Core.hpp"
#include "ANN_CoreCPU.hpp"
#include "ANN_CoreGPU.hpp"

#include <OCLW_Core.hpp>
#include <QFile>

using namespace ANN;

//===================================================================================================================//

template <typename T>
Core<T>::Core(const CoreConfig<T>& coreConfig) {
  this->coreTypeType = coreConfig.coreTypeType;
  this->coreModeType = coreConfig.coreModeType;

  this->layersConfig = coreConfig.layersConfig;
  this->trainingConfig = coreConfig.trainingConfig;
  this->parameters = coreConfig.parameters;
}

//===================================================================================================================//

template <typename T>
std::unique_ptr<Core<T>> Core<T>::makeCore(const CoreConfig<T>& coreConfig) {
  if (coreConfig.coreTypeType == CoreTypeType::CPU) {
    return std::make_unique<CoreCPU<T>>(coreConfig);
  } else {
    return std::make_unique<CoreGPU<T>>(coreConfig);
  }
}

//===================================================================================================================//

template <typename T>
void Core<T>::sanityCheck(const CoreConfig<T>& coreConfig) {
  if (coreConfig.coreTypeType == CoreTypeType::UNKNOWN) {
    throw std::runtime_error("Unkown coreTypeType");
  }

  if (coreConfig.coreModeType == CoreModeType::UNKNOWN) {
    throw std::runtime_error("Unkown coreModeType");
  }
}

//===================================================================================================================//

// (Optional) Explicit template instantiations.
template class ANN::Core<int>;
template class ANN::Core<double>;
template class ANN::Core<float>;
