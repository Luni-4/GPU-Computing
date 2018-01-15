#ifdef _WIN32
#include "Windows.h"
#endif

#include "Common.h"

// Librerie di progetto
#include "Mnist.h"
#include "Cifar.h"
#include "FullyConnected.h"
#include "FullyConnectedStream.h"
#include "Convolutional.h"
#include "ConvolutionalStream.h"
#include "Network.h"

#ifdef DEBUG
//#include "Test.h"
#endif

#ifdef TOYINPUT
#include "ToyInput.h"
#endif

int main() {

#ifdef DEBUG
	//getTime(&test_mnist_input, "test_mnist_input");

	//getTime(&test_cifar10_input, "test_cifar10_input");

	//getTime(&test_cifar100_input, "test_cifar100_input");

	//test_fully();
#endif

	// Leggere i dati
#ifdef _WIN32
	std::unique_ptr<Data> d(new Mnist("../data/mnist/"));
#else

#ifdef TOYINPUT
	std::unique_ptr<Data> d(new ToyInput());
#else
	std::unique_ptr<Data> d(new Mnist("data/mnist/"));
#endif

#endif

	// Vettore contenente i livelli della rete
	std::vector<std::unique_ptr<LayerDefinition>> layers;

	// Inizializzare i livelli
#ifdef _WIN32
	//dim_filtro, n_filtri, stride
	layers.emplace_back(new Convolutional(5, 1, 1, SIGMOID));
	//layers.emplace_back(new Convolutional(7, 1, 1, SIGMOID));
	//layers.emplace_back(new Convolutional(3, 1, 1, SIGMOID));
	//layers.emplace_back(new ConvolutionalStream(5, 1, 1, SIGMOID));
	//layers.emplace_back(new ConvolutionalStream(5, 1, 1, SIGMOID));
	//layers.emplace_back(new Batch(5, 1, 1));
	//layers.emplace_back(new Batch(5, 1, 1));
	//layers.emplace_back(new FullyConnected(100, SIGMOID));
	layers.emplace_back(new FullyConnected(300, SIGMOID));
	layers.emplace_back(new FullyConnected(10, SIGMOID));
	//layers.emplace_back(new FullyConnected_Stream(500, SIGMOID));
	//layers.emplace_back(new FullyConnectedStream(300, SIGMOID));
	//layers.emplace_back(new FullyConnectedStream(10, SIGMOID));

	// MEMO: learning rate base 0.001
#else
	layers.emplace_back(new FullyConnected(300, SIGMOID));
	//layers.emplace_back(new Convolutional(5, 1, 1, SIGMOID));
	//layers.emplace_back(new Convolutional(5, 1, 1, SIGMOID));
	layers.emplace_back(new FullyConnected(10, SIGMOID));
	//layers.emplace_back(new FullyConnected_Stream(500, SIGMOID));
	//layers.emplace_back(new FullyConnected_Stream(300, SIGMOID));
#endif

		// Creare la rete
	Network nn(layers);

	//#ifdef DEBUG
	auto start = std::chrono::high_resolution_clock::now();
	//#endif

	//std::cout.precision(64);
	double learningRate = 0.208;
	std::cout << "learningRate:" << learningRate << std::endl;

	// Training
	nn.train(d.get(), 1, learningRate);

	//#ifdef DEBUG
	auto finish = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(finish - start);
	std::cout << "Tempo di esecuzione della funzione di train: " << elapsed.count() << std::endl;
	//#endif

	//nn.printW();
	// Stampa i pesi prodotti dalla rete su un file
	//nn.printWeightsOnFile("Weights.txt");

	// Test
	nn.predict(d.get());

#ifdef _WIN32
	system("pause");
#endif
}
