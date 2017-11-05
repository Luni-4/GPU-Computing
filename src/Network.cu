// Cuda
#include <cuda_runtime.h>

#include <iostream>

#include "Common.h"
#include "Network.h"

Network::Network(const std::vector<std::unique_ptr<LayerDefinition>> &layers) {
    
    for(auto& l: layers)
		_layers.push_back(l.get());
}

Network::~Network() {    
    // Liberare la memoria del device
    CHECK(cudaFree(inputImg));
    //CHECK(cudaDeviceReset());
}

void Network::train(Data *data, const int &epoch, const double &eta, const double &lambda) {
    //Leggere i dati dal training set
    data->readTrainData();
    
    // Caricare i dati in Cuda
    cudaDataLoad(data);
    
    // Inizializzare le strutture della rete
    cudaInitStruct(data);
    
    // Numero di esempi nel training set
    const int nImages = data->getLabelSize();
    
    // Dimensione della singola immagine
    const int imgDim = data->getImgDimension();
    
    // Quantità di dati da allocare e copiare
    const int iBytes = imgDim * sizeof(double); 
	
	// Allocare il buffer di input della singola coppia (etichetta,immagine)
	CHECK(cudaMalloc((void**)&inputImg, iBytes));
	
	int i = 0;
    
    // Elabora ogni immagine
    //for(int i = 0; i < nImages; i++)
    //{
        int imgIndex = i * imgDim;
        
        // Copia dell'immagine corrente nel buffer
        CHECK(cudaMemcpy(inputImg, (cudaData + imgIndex), iBytes, cudaMemcpyDeviceToDevice));        
         
        //for(int j = 0; j < epoch; j++)
        //{
            // Forward_propagation per ogni livello
            forwardPropagation();
            
            // Calcolo dell'errore per ogni livello
            //error();
            
            // Backward_propagation per ogni livello
           // backwardPropagation();
        //}   
    
    //}
    
    // cancellare i dati di train dal device
    CHECK(cudaFree(cudaData));
    CHECK(cudaFree(cudaLabels));
    
}


void Network::cudaDataLoad(Data *data) {
	const int dBytes = data->getDataSize() * sizeof(double);
	const int lBytes = data->getLabelSize() * sizeof(uint8_t);

	// Allocare le matrici
	CHECK(cudaMalloc((void**)&cudaData, dBytes));
	CHECK(cudaMalloc((void**)&cudaLabels, lBytes));
	
	// Passare i dati
	CHECK(cudaMemcpy(cudaData, data->getCudaData(), dBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(cudaLabels, data->getCudaLabels(), lBytes, cudaMemcpyHostToDevice));

	// Liberare i dati dalla CPU
	data->clearDataCPU();
}

void Network::cudaInitStruct(Data *data) {
    
    _layers[0]->defineCuda(data->getImgWidth(), data->getImgHeight(), data->getImgDepth());
    
    for(std::size_t i = 1; i < _layers.size(); i++)
    {
        const int prevWidth = _layers[i-1]->getWidth();
        const int prevHeight = _layers[i-1]->getHeight();
        const int prevDepth = _layers[i-1]->getDepth();
        
        _layers[i]->defineCuda(prevWidth, prevHeight, prevDepth);
    } 
}

void Network::forwardPropagation(){

    _layers[0]->forward_propagation(inputImg);   
    
    
    /*for(std::size_t i = 1; i < _layers.size(); i++)
    {       
        _layers[i]->forward_propagation(_layers[i-1].getOutput());
    }*/
}
