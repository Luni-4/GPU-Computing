#include <cstdio>
#include <iostream>

// Librerie Cuda

// Librerie di progetto
#include "Mnist.h"
#include "FullyConnected.h"

int main() {

    // Leggere i dati
    Data* d = new Mnist("../data/");       
    
    // Lettura dati di training
    d->readTrainData(); 
    
    // Vettori di valori per Cuda    
    const double* m = d->getData();
    const uint8_t* s = d->getLabels(); 
    
    delete d;
    
    // Creare i layer
    //std::vector<LayerDefinition*> layers(1);
    
    //layers[0] = new ConvolutionalLayer(rng, 768, 300);
    LayerDefinition* layer = new FullyConnected(10, RELU);
    
    printf("%d %d %d %d",layer->getWidth(), layer->getHeight(), layer->getLayerType(), layer->getActivationFunction());
    
    layer->defineCuda(28,28,1);
    
    delete layer;
    
    // Creare la rete
    /*Network nn(layers, // param);
    
    // Training
    nn.train(//param);
    
    // Test
    nn.predict(//param);
    
    // Cancellare i layer
    for (int i = 0; i < layers.size(); i++) {
        delete layers[i];
    }*/

}
