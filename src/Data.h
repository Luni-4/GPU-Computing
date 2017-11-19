#pragma once

#include <vector>
#include <string>

class Data {

public:
    
    Data() {}    
    virtual ~Data() {}
    
    // Impedisce che vengano fatte copie e assegnamenti alla classe
    Data(Data const&) = delete;
    Data& operator=(Data const&) = delete;
    
    virtual void readTrainData(void) = 0;
    virtual void readTestData(void) = 0;
    
    virtual uint32_t getImgWidth(void) const = 0;
    virtual uint32_t getImgHeight(void) const = 0;
    virtual uint32_t getImgDepth(void) const = 0;
    
    virtual uint32_t getImgDimension(void) const = 0;
    
    inline const double* getData(void) const { return data.data(); }
    inline const uint8_t* getLabels(void) const { return labels.data(); }
    
    inline size_t getDataSize(void) const { return data.size(); }
    inline size_t getLabelSize(void) const { return labels.size(); }
    
    void clearData(void) {
        data.clear();
    }
    
    void clearLabels(void) {    
        labels.clear();
    }
    
protected:
    std::vector<double> data = {};
    std::vector<uint8_t> labels = {};
    
};
