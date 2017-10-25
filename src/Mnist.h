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
    
    
private:
    void readImages(const std::string& datafile);
    void readLabels(const std::string& datafile);
    
    inline uint32_t flipBytes(const uint32_t& n);

private:
    const std::string _filename;
    
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
    uint32_t maxLabels;
    
    bool isTrain;
    bool isTest;
};


