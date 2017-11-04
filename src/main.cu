#include <cstdio>
#include <iostream>

// Librerie Cuda

// Librerie di progetto
#include "Mnist.h"
#include "FullyConnected.h"



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

	test_input();

	test_fully();

	// Creare i layer
	//std::vector<LayerDefinition*> layers(1);

	//layers[0] = new ConvolutionalLayer(rng, 768, 300);

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

#if _WIN32
	system("pause");
#endif

}
