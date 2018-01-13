#include <cstdio>
#include <ctime>

#include "educnn.h"

int main(int argc, char** argv) {
    Matrix train_data = mnist::train_data();
    Matrix train_label = mnist::train_label();

    Random* rng = new Random((unsigned int)time(0));    
   

    std::vector<AbstractLayer*> layers(3);
    layers[0] = new ConvolutionLayer(rng, Size(28, 28), Size(5, 5), 1, 5);
    layers[1] = new ConvolutionLayer(rng, Size(24, 24), Size(3, 3), 5, 5);
    layers[2] = new FullyConnectedLayer(rng, 2420, 10);  
    

    Network network(layers, train_data, train_label, 1);    

    Timer timer;
    timer.start();
    network.train(1, 0.1, 0);
    printf("Time: %f sec\n", timer.stop());

    Matrix test_data = mnist::test_data();
    Matrix test_label = mnist::test_label();

    Matrix pred = network.predict(test_data);

    double ratio = check(pred, test_label);
    printf("Ratio: %f %%\n", ratio);

    delete rng;
    for (int i = 0; i < layers.size(); i++) {
        delete layers[i];
    }
}
