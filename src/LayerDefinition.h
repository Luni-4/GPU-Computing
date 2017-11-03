#pragma once

typedef enum LayerType {CONVOLUTIONAL, FULLY_CONNECTED} LayerType;
typedef enum ActFctType {SIGMOID, TANH, RELU, NONE} ActFctType;

class LayerDefinition
{

public:

    LayerDefinition(const int &width, const int &height, const int &depth, const LayerType &l, const ActFctType &a):
    	_width(width),
    	_height(height),
    	_depth(depth),
    	_l(l),
    	_a(a) {}
    virtual ~LayerDefinition() {}
    
    // Impedisce che vengano fatte copie e assegnamenti alla classe
    LayerDefinition(LayerDefinition const&) = delete;
    LayerDefinition& operator=(LayerDefinition const&) = delete;   
    
    virtual int getLayerNodeCount() = 0; // Numero di nodi del livello
    virtual int getWeightCount(const int &prevLayerNode) = 0; // Numero di pesi del livello (dipendono dal livello precedente)
    virtual std::vector<double> getWeight() = 0; // Restituisce da Cuda il vettore di pesi
    virtual std::vector<double> getBias() = 0; // Restituisce da Cuda il vettore di pesi

    virtual void forward_propagation() = 0; // Definita in Cuda
    
    virtual void back_propagation() = 0; // Aggiorno i pesi di questo livello in Cuda
    
    virtual void defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) = 0; // Definisce le strutture di Cuda e le inizializza
    
    inline LayerType getLayerType() const { return _l; }
    inline int getWidth() const { return _width; }
    inline int getHeight() const { return _height; }
    inline int getDepth() const { return _depth; }
    inline ActFctType getActivationFunction() const { return _a; }
    
    
protected:
    LayerType _l;
    ActFctType _a;
    int _width;
    int _height;
    int _depth;
    
};
