#include <cstdio>
#include <iostream>

// Librerie Cuda
#include <cuda_runtime.h>

// Librerie di progetto
#include "Mnist.h"

int main() {

    // Leggere i dati
    Data* d = new Mnist("../data/");    
    
    // Lettura dati di training
    d->readTrainData();    
    
    // Vettori di valori per Cuda    
    const double * m = d->getData();
    const uint8_t* s = d->getLabels(); 
    
    delete d;
    
    /*// Creare i layer
    std::vector<LayerDefinition*> layers(2);
    
    layers[0] = new ConvolutionalLayer(rng, 768, 300);
    layers[1] = new OutputLayer(rng, 300, 10);
    
    // Creare la rete
    Network nn(layers, // param);
    
    // Training
    nn.train(//param);
    
    // Test
    nn.predict(//param);
    
    // Cancellare i layer
    for (int i = 0; i < layers.size(); i++) {
        delete layers[i];
    }*/

}
