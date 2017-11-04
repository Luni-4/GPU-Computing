#include <cstdio>
#include <iostream>

// Librerie Cuda

// Librerie di progetto
#include "Mnist.h"
#include "FullyConnected.h"
#include "Network.h"



void test_input() {
	// Leggere i dati
	Data* d = new Mnist("../data/");

	// Lettura dati di training
	d->readTrainData();

	delete d;
}


void test_fully() {
	LayerDefinition* layer = new FullyConnected(10, RELU);

	printf("%d %d %d %d\n", layer->getWidth(), layer->getHeight(), layer->getLayerType(), layer->getActivationFunction());

	layer->defineCuda(28, 28, 1);

	std::vector<double> p = layer->getWeight();

	std::cout << p.size() << std::endl;

	for (auto t : p)
		std::cout << t << std::endl;

	delete layer;
}

int main() {

	//test_input();

	//test_fully();
	
	// Leggere i dati
	Data* d = new Mnist("../data/");

	// Creare i layer
	std::vector<LayerDefinition*> layers(1);

	layers[0] = new FullyConnected(10, RELU);

	// Creare la rete
	Network nn(layers);

	// Training
	nn.train(d, 20, 0.5, 0.1);

	// Test
	//nn.predict(//param);

	// Cancellare i layer
	for (std::size_t i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
	
	delete d;

#ifdef _WIN32
	system("pause");
#endif

}
