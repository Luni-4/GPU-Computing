#pragma once

#include <cstdio>
#include <vector>
#include <cstdint>

class Data {

protected:
    std::vector<double> data = {};
    std::vector<uint8_t> labels = {};

public:
    
    Data() {}
    virtual ~Data() {}
    
     // Impedisce che vengano fatte copie e assegnamenti alla classe
    Data(Data const&) = delete;
    Data& operator=(Data const&) = delete;
    
    virtual void readTrainData() = 0;
    virtual void readTestData() = 0;
    
    inline const double* getData() const { return &data[0]; }
    inline const uint8_t* getLabels() const { return &labels[0]; }
};
