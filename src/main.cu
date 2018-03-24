#ifdef _WIN32
#include "Windows.h"
#endif

#include "Common.h"

// Librerie di progetto
#include "Mnist.h"
#include "Cifar.h"

#include "FullyConnected.h"
#include "FullyConnectedStream.h"
#include "ConvolutionalEDUCNN.h"
#include "Network.h"

#ifdef DEBUG
#include "Test.h"
#endif

#ifdef TOYINPUT
#include "ToyInput.h"
#endif

int main() {

#ifdef DEBUG
	getTime(&test_mnist_input, "test_mnist_input");

	getTime(&test_cifar10_input, "test_cifar10_input");

	getTime(&test_cifar100_input, "test_cifar100_input");

	test_fully();
#endif

	// Leggere i dati
#ifdef _WIN32
	std::unique_ptr<Data> d(new Mnist("../data/mnist/"));
	//std::unique_ptr<Data> d(new Cifar("../data/cifar/cifar10/"));
#else

#ifdef TOYINPUT
	std::unique_ptr<Data> d(new ToyInput());
#else
	std::unique_ptr<Data> d(new Mnist("data/mnist/"));
	//std::unique_ptr<Data> d(new Cifar("data/cifar/cifar10/"));
#endif

#endif

	// Vettore contenente i livelli della rete
	std::vector<std::unique_ptr<LayerDefinition>> layers;

	int depth = 1;

	// Inizializzare i livelli
#ifdef _WIN32	
	//dim_filtro, n_filtri, stride
	
	layers.emplace_back(new ConvolutionalEDUCNN(5, depth, 1));
	//layers.emplace_back(new FullyConnected(400, SIGMOID));
	layers.emplace_back(new ConvolutionalEDUCNN(5, depth, 1));
	//layers.emplace_back(new FullyConnected(100, SIGMOID));
	layers.emplace_back(new ConvolutionalEDUCNN(5, depth, 1));
	//layers.emplace_back(new FullyConnected(300, SIGMOID));
	layers.emplace_back(new FullyConnected(10, SIGMOID));

#else
	//layers.emplace_back(new ConvolutionalEDUCNN(5, depth, 1));
	//layers.emplace_back(new ConvolutionalEDUCNN(5, depth, 1));
	layers.emplace_back(new FullyConnected(300, SIGMOID));
	layers.emplace_back(new FullyConnected(10, SIGMOID));
#endif

		// Creare la rete
		Network nn(layers);

		auto start = std::chrono::high_resolution_clock::now();

		//std::cout.precision(64);
		double learningRate = 1;

		int epoch = 1;
		std::cout << "\nlearningRate:" << learningRate << std::endl;

		// Training
		nn.train(d.get(), epoch, learningRate);

		auto finish = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(finish - start);
		std::cout << "Tempo di esecuzione della funzione di train: " << elapsed.count() << std::endl;
		
		// Test
		nn.predict(d.get());

#ifdef _WIN32
	system("pause");
#endif
}
