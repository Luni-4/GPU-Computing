#pragma once

#include "Data.h"

const std::string train_image_file_mnist = "train-images-idx3-ubyte";
const std::string train_label_file_mnist = "train-labels-idx1-ubyte";
const std::string test_image_file_mnist  = "t10k-images-idx3-ubyte";
const std::string test_label_file_mnist  = "t10k-labels-idx1-ubyte";

const uint32_t nImages_m = 60000;
const uint32_t imgWidth_m = 28;
const uint32_t imgHeight_m = 28; 

class Mnist : public Data {

public:
    Mnist(const std::string &filePath);
    virtual ~Mnist();
    
    void readTrainData(void) override;
    void readTestData(void) override;
    
    uint32_t getImgWidth(void) const override { return imgWidth_m; }
    uint32_t getImgHeight(void) const override { return imgHeight_m; }
    uint32_t getImgDepth(void) const override { return 1; }
    uint32_t getImgDimension(void) const override { return _imgDim; }
    
    
private:
    void readImages(const std::string &fileName);
    void readLabels(const std::string &fileName);
    
    inline void cleanSetData(void);

private:
    const std::string _filePath;
    
    uint32_t _imgDim;   
    
    bool _isTrain;
    bool _isTest;
};


