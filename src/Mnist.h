#pragma once

#include "Data.h"

#include <string>


const std::string train_image_file_mnist = "train-images-idx3-ubyte";
const std::string train_label_file_mnist = "train-labels-idx1-ubyte";
const std::string test_image_file_mnist  = "t10k-images-idx3-ubyte";
const std::string test_label_file_mnist  = "t10k-labels-idx1-ubyte";
 

class Mnist : public Data {

public:
    Mnist(const std::string filename);
    ~Mnist();
    void readTrainData() override;
    void readTestData() override;
    
    uint32_t getImgWidth() const { return imgWidth; }
    uint32_t getImgHeight() const { return imgHeight; }
    uint32_t getImgDepth() const { return 1; }
    
    
private:
    void readImages(const std::string& datafile);
    void readLabels(const std::string& datafile);
    
    inline uint32_t flipBytes(const uint32_t& n);

private:
    const std::string _filename;
    
    uint32_t imgWidth;
    uint32_t imgHeight;    
    
    bool isTrain;
    bool isTest;
};


