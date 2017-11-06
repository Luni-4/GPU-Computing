#include <cstdio>
#include <iostream>
#include <memory>

// Librerie Cuda

// Librerie di progetto
#include "Mnist.h"
#include "FullyConnected.h"
#include "Convolutional.h"
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

	std::vector<double> w = layer->getWeights();

	std::cout << w.size() << std::endl;

	for (auto t : w)
		std::cout << t << std::endl;

	delete layer;
}

int main() {

	//test_input();

	//test_fully();

	// Leggere i dati
	std::unique_ptr<Data> d(new Mnist("../data/"));

	// Vettore contenente i livelli della rete
	std::vector<std::unique_ptr<LayerDefinition>> layers;

	// Inizializzare i livelli	
	//layers.emplace_back(new FullyConnected(10, RELU));
	layers.emplace_back(new Convolutional(5, 1, 1, RELU));

	// Creare la rete
	Network nn(layers);

	// Training
	nn.train(d.get(), 20, 0.5, 0.1);

	// Test
	//nn.predict(//param);

#ifdef _WIN32
	system("pause");
#endif

}
