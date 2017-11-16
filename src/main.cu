#include <iostream>

#ifdef _WIN32
#include "Windows.h"
#endif

#include "Common.h"

// Librerie di progetto
#include "Mnist.h"
#include "FullyConnected.h"
#include "Convolutional.h"
#include "Network.h"

#ifdef DEBUG
#include "Test.h"
#endif

int main() {

#ifdef DEBUG
	test_mnist_input();
	
	test_cifar_input();

	test_fully();
#endif

	// Leggere i dati
	std::unique_ptr<Data> d(new Mnist("data/mnist/"));

	// Vettore contenente i livelli della rete
	std::vector<std::unique_ptr<LayerDefinition>> layers;

	// Inizializzare i livelli	
#ifdef _WIN32	
	layers.emplace_back(new Convolutional(5, 1, 1, RELU));
#else
	layers.emplace_back(new FullyConnected(4, SIGMOID));
	layers.emplace_back(new FullyConnected(3, SIGMOID));
	layers.emplace_back(new FullyConnected(2, SIGMOID));
#endif

	// Creare la rete
	Network nn(layers);

	// Training
	nn.train(d.get(), 20, 0.001);
	
	// Stampa i pesi prodotti dalla rete su un file
	nn.printWeightsOnFile("Weights.txt");

	// Test
	nn.predict(d.get());
	
	// Array contenente le predizioni
	std::vector<uint8_t> predictions = nn.getPredictions();
	
	// Errore commesso dalla rete sul test set
	int error = nn.getTestError();
	
	// Stampare le predizioni
	printLabels(predictions);
	
	// Stampare l'errore
	std::cout << std::endl << std::endl << error << std::endl;	

#ifdef _WIN32
	system("pause");
#endif

}
