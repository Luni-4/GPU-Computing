#pragma once

#include "Data.h"


const std::string train_image_file_mnist = "train-images-idx3-ubyte";
const std::string train_label_file_mnist = "train-labels-idx1-ubyte";
const std::string test_image_file_mnist  = "t10k-images-idx3-ubyte";
const std::string test_label_file_mnist  = "t10k-labels-idx1-ubyte";
 

class Mnist : public Data {

public:
    Mnist(const std::string &filename);
    virtual ~Mnist();
    
    void readTrainData(void) override;
    void readTestData(void) override;
    
    uint32_t getImgWidth(void) const override { return _imgWidth; }
    uint32_t getImgHeight(void) const override { return _imgHeight; }
    uint32_t getImgDepth(void) const override { return 1; }
    uint32_t getImgDimension(void) const override { return _imgWidth * imgHeight; }
    
    
private:
    void readImages(const std::string &datafile);
    void readLabels(const std::string &datafile);
    
    inline uint32_t flipBytes(const uint32_t &n);

private:
    const std::string _filename;
    
    uint32_t _imgWidth;
    uint32_t _imgHeight;    
    
    bool _isTrain;
    bool _isTest;
};


