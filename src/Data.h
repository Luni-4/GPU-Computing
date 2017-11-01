#pragma once

#include <cstdio>
#include <vector>

class Data {

public:
    
    Data() {}
    virtual ~Data() {}
    
    // Impedisce che vengano fatte copie e assegnamenti alla classe
    Data(Data const&) = delete;
    Data& operator=(Data const&) = delete;
    
    virtual void readTrainData() = 0;
    virtual void readTestData() = 0;
    
    virtual uint32_t getImgWidth() const = 0;
    virtual uint32_t getImgHeight() const = 0;
    virtual uint32_t getImgDepth() const = 0;
    
    
    inline const double* getData() const { return &data[0]; }
    inline const uint8_t* getLabels() const { return &labels[0]; }
    inline size_t getDataSize() const { return data.size(); }
    inline size_t getLabelSize() const { return labels.size(); }
    
protected:
    std::vector<double> data = {};
    std::vector<uint8_t> labels = {};
};
