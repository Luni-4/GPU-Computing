#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "LayerDefinition.h"

class FullyConnected: public LayerDefinition
{

public:

    FullyConnected(const int &width, const int &height, const ActFctType &a);
    FullyConnected(const int &width, const ActFctType &a); 
    ~FullyConnected();
    
    int getLayerNodeCount() override;
    int getWeightCount(const int &prevLayerNode) override;
    std::vector<double> getWeight() override;
    
    void forward_propagation() override;
    
    void back_propagation() override;
    
    void defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) override;
    
    
private:
    int _wBytes;
    bool _isCuda;

    double *weight; // Matrice dei pesi in Cuda
    double *bias; // Matrice per i bias in Cuda
    double *output; // Matrice dell'output in Cuda
    double *error; // Matrice degli errori
    
};
