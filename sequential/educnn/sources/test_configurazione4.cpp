#include <cstdio>
#include <ctime>

#include "educnn.h"

int main(int argc, char** argv) {
    Matrix train_data = mnist::train_data();
    Matrix train_label = mnist::train_label();

    Random* rng = new Random((unsigned int)time(0));    
   

    std::vector<AbstractLayer*> layers(3);
    layers[0] = new ConvolutionLayer(rng, Size(28, 28), Size(5, 5), 1, 1);
    layers[1] = new FullyConnectedLayer(rng, 576, 300);
    layers[2] = new FullyConnectedLayer(rng, 300, 10); 
    

    Network network(layers, train_data, train_label, 1);    

    Timer timer;
    double learning = 0.01;
    int i = 1;
    do
    {
        printf("Configurazione: %d\n\n",i);
        timer.start();
        network.train(1, learning, 0);
        printf("Time: %f sec\n", timer.stop());

        Matrix test_data = mnist::test_data();
        Matrix test_label = mnist::test_label();

        Matrix pred = network.predict(test_data);

        double ratio = check(pred, test_label);
        printf("Ratio: %f %%\n\n", ratio);
        
        learning += 0.05;
        i++;
    }while(learning < 0.8);

    delete rng;
    for (int i = 0; i < layers.size(); i++) {
        delete layers[i];
    }
}
