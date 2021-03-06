#pragma once

inline void printDataInformation(Data *d) {

	// Informazioni sui dati
	/*std::cout << "Width: " << d->getImgWidth() << std::endl;
	std::cout << "Height: " << d->getImgHeight() << std::endl;
	std::cout << "Depth: " << d->getImgDepth() << std::endl;
	std::cout << "Image Dimension: " << d->getImgDimension() << std::endl;

	std::cout << std::endl;*/

	auto dataSize = d->getDataSize();
	auto labelSize = d->getLabelSize();

	std::cout << "Data Dimension: " << dataSize << std::endl;
	std::cout << "Label Dimension: " << labelSize << std::endl;

	//printInputData(d->getData(), dataSize);

	//std::cout << std::endl << std::endl;

	//printInputLabels(d->getLabels(), labelSize);	
}

void test_mnist_input() {
	// Leggere i dati
	Data* d = new Mnist("data/mnist/");

	// Lettura dati di training
	d->readTrainData();

	delete d;
}

void test_cifar10_input() {
	// Leggere i dati
	Data* d = new Cifar("data/cifar/cifar10/");

	// Lettura dati di training
	d->readTrainData();

	printDataInformation(d);

	delete d;
}

void test_cifar100_input() {
	// Leggere i dati
	Data* d = new Cifar("data/cifar/cifar100/", false);

	// Lettura dati di training
	d->readTrainData();

	//printDataInformation(d);

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

