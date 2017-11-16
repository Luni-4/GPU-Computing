#pragma once

#include "Data.h"

/* Cifar 10 */

// Il training set è suddivisio in 5 file binari da 10000 esempi ciascuno (50000 esempi di training)
const std::vector<std::string> train_cifar10 = {"data_batch_1.bin",
                                                "data_batch_2.bin",
                                                "data_batch_3.bin",
                                                "data_batch_4.bin",
                                                "data_batch_5.bin" };

// Il test set è composto da 10000 esempi
const std::string test_cifar10 = "test_batch.bin";


/* Cifar 100 */

// Il training set si compone di 50000 esempi
const std::string train_cifar100 = "train.bin";

// Il test set è composto da 10000 esempi
const std::string test_cifar100 = "test.bin";


const int cifarTrainDim = 50000;
const int cifarTestDim = 10000;
 

class Cifar : public Data {

public:
    Cifar(const std::string &filename, const bool &isCifar10);
    virtual ~Cifar();
    
    void readTrainData(void) override;
    void readTestData(void) override;
    
    uint32_t getImgWidth(void) const override { return _imgWidth; }
    uint32_t getImgHeight(void) const override { return _imgHeight; }
    uint32_t getImgDepth(void) const override { return _imgDepth; }
    uint32_t getImgDimension(void) const override { return _imgDim; }
    
    
private:
    inline void readCifarTrain10(const std::vector<std::string> &datafile);
    
    void readCifar10(const std::string &datafile);
    void readCifar100(const std::string &datafile, const int &iterations);
    
    inline void cleanSetData(const int &set);

private:
    const std::string _filename;
    std::vector<uint8_t> _pixel;
    
    uint32_t _imgWidth;
    uint32_t _imgHeight;
    uint32_t _imgDepth;
    uint32_t _imgDim;
    
    bool _isCifar10;   
    
    bool _isTrain;
    bool _isTest;
};


