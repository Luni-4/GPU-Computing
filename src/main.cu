#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>

// Librerie Cuda

#ifdef _WIN32
#include "Windows.h"
#endif

#ifdef DEBUG
#include "Common.h"
#endif

// Librerie di progetto
#include "Mnist.h"
#include "FullyConnected.h"
#include "Convolutional.h"
#include "Network.h"

#ifdef DEBUG

void test_input() {
	// Leggere i dati
	Data* d = new Mnist("../data/");

	// Lettura dati di training
	d->readTrainData();

	delete d;
}


void test_fully() {
	LayerDefinition* layer = new FullyConnected(10, RELU);

	// Stampa dei vari parametri del livello
	printf("Larghezza layer: %d\n", layer->getWidth());
	printf("Altezza layer: %d\n", layer->getHeight());
	printf("Tipo di layer: %d\n", layer->getLayerType());
	printf("Funzione di attivazione: %d", layer->getActivationFunction());

	// Definizione di un'immagine di 4 pixel rgb
	layer->defineCuda(2, 2, 3);

	double *inp;

	CHECK(cudaMalloc((void**)&inp, 12 * sizeof(double)));

	CHECK(cudaMemset(inp, 0, 12 * sizeof(double)));

	std::cout << "\n\n\nImmagine di input RGB\n\n";
	printFromCuda(inp, 12);

	// Passare l'immagine ed eseguire prodotto più aggiunta del bias
	layer->forward_propagation(inp);

	std::vector<double> w = layer->getWeights();

	std::cout << "\n\n\nNumero dei pesi: ";
	std::cout << w.size();
	std::cout << "\n\nPesi\n\n";

	for (auto t : w)
		std::cout << t << std::endl;

	layer->deleteCuda();

	CHECK(cudaFree(inp));

	delete layer;
}

#endif

int main() {

#ifdef DEBUG
	//test_input();

	//test_fully();
#endif

	// Leggere i dati
	std::unique_ptr<Data> d(new Mnist("../data/"));

	// Vettore contenente i livelli della rete
	std::vector<std::unique_ptr<LayerDefinition>> layers;

	// Inizializzare i livelli	
#ifdef _WIN32	
	layers.emplace_back(new FullyConnected(10, RELU));
	//layers.emplace_back(new Convolutional(5, 1, 1, RELU));
#else
	layers.emplace_back(new FullyConnected(10, RELU));
#endif

	// Creare la rete
	Network nn(layers);

	// Training
	nn.train(d.get(), 20, 0.5);

	// Test
	//nn.predict(//param);

#ifdef _WIN32
	system("pause");
#endif

}
