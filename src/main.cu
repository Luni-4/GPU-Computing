#ifdef _WIN32
#include "Windows.h"
#endif

#include "Common.h"

// Librerie di progetto
#include "Mnist.h"
#include "Cifar.h"
#include "FullyConnected.h"
//#include "Convolutional.h"
#include "Network.h"

#ifdef DEBUG
#include "Test.h"
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
	//layers.emplace_back(new Convolutional(5, 1, 1, RELU));
	//layers.emplace_back(new FullyConnected(10, SIGMOID));
#else
	layers.emplace_back(new FullyConnected(300, SIGMOID));
	//layers.emplace_back(new FullyConnected(10, SIGMOID));
#endif

	// Creare la rete
	Network nn(layers);
	
//#ifdef DEBUG
    auto start = std::chrono::high_resolution_clock::now();  
//#endif

	// Training
	nn.train(d.get(), 1, 0.1);

//#ifdef DEBUG
    auto finish = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(finish - start);
    std::cout << "Tempo di esecuzione della funzione di train: " << elapsed.count() << std::endl;
//#endif
    

	// Stampa i pesi prodotti dalla rete su un file
	//nn.printWeightsOnFile("Weights.txt");

	// Test
	/*nn.predict(d.get());

	// Array contenente le predizioni
	std::vector<uint8_t> predictions = nn.getPredictions();

	// Stampare le predizioni
	printLabels(predictions);*/	
	
#ifdef _WIN32
	system("pause");
#endif

}
