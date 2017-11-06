#include <cstdio>
#include <ctime>

#include "educnn.h"

int main(int argc, char** argv) {
    Matrix train_data = mnist::train_data();
    Matrix train_label = mnist::train_label();

    Random* rng = new Random((unsigned int)time(0));
    
    /* 
    
    I valori dei livelli convoluzionali non sono uguali a quelli della dnn perché in questa rete lo stride è fisso a 1, mentre nella dnn
    lo stride della rete di esempio è di 2 (calcolato dalla funzione calcStride in dnn.c). 
    
    In nessuna delle due reti è implementato il meccanismo di padding.
    
    Per calcolare le dimensioni dei vari livelli convoluzionali sto usando la formula dell'articolo che spiega le reti neurali convoluzionali,
    pagina 2. Viene usata la funzione di ceil per evitare di perdere un pixel, a causa del funzionamento della divisione tra interi
    
    DNN
    
    L'output del primo livello convoluzionale è uguale a 13 = ceil( (28 - 5) / 2 ) + 1 
    L'output del secondo livello convoluzionale è uguale a 6 = ceil( (13 - 3) / 2 ) + 1
    
    EDUCNN
    
    Lo stride è 1 e la ceil non è più necessaria
    
    L'output del primo livello convoluzionale è uguale a 24 = (28 - 5) + 1 
    L'output del secondo livello convoluzionale è uguale a 22 = (24 - 3) + 1
    
    Calcolo dell'input del FullyConnectedLayer (corrispondente al livello di output della dnn)
    
    Numero di nodi livello precedente * depth livello precedente = (22 * 22) * 5 = 2420    
    
    */

    std::vector<AbstractLayer*> layers(3);
    layers[0] = new ConvolutionLayer(rng, Size(28, 28), Size(5, 5), 1, 5);
    layers[1] = new ConvolutionLayer(rng, Size(24, 24), Size(3, 3), 5, 5);
    layers[2] = new FullyConnectedLayer(rng, 2420, 10);  
    

    Network network(layers, train_data, train_label, 50);
    

    Timer timer;
    timer.start();
    network.train(20, 0.1, 0.5);
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
