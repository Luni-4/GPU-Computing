#pragma once

#include "Data.h"

class ToyInput : public Data {

public:
    ToyInput();
    virtual ~ToyInput();
    
    void readTrainData(void) override;
    void readTestData(void) override;
    
    uint32_t getImgWidth(void) const override { return _imgWidth; }
    uint32_t getImgHeight(void) const override { return _imgHeight; }
    uint32_t getImgDepth(void) const override { return _imgDepth; }
    uint32_t getImgDimension(void) const override { return _imgDim; }
    
    
private:    
    inline void cleanSetData(const uint32_t &nImages);
    inline void process(void);

private:
    uint32_t _imgWidth;
    uint32_t _imgHeight;
    uint32_t _imgDepth;   
    uint32_t _imgDim;   
    
    bool _isTrain;
    bool _isTest;
};


