#pragma once

typedef enum LayerType { CONVOLUTIONAL, FULLY_CONNECTED } LayerType;
typedef enum ActFctType { SIGMOID, TANH, RELU } ActFctType;

class LayerDefinition {

public:

	LayerDefinition(const int &width, const int &height, const int &depth, const LayerType &l, const ActFctType &a) :
		_width(width),
		_height(height),
		_depth(depth),
		_l(l),
		_a(a) {
	}
	virtual ~LayerDefinition() {}

	// Impedisce che vengano fatte copie e assegnamenti alla classe
	LayerDefinition(LayerDefinition const&) = delete;
	LayerDefinition& operator=(LayerDefinition const&) = delete;

	virtual int getNodeCount() const = 0;
	virtual int getWeightCount(const int &prevLayerNode) const = 0;
	virtual std::vector<double> getWeights() = 0;
	virtual std::vector<double> getBias() = 0;

	virtual void forward_propagation(const double *prevOutput) = 0;

	virtual void back_propagation(const double *prevOutput, const double *forwardWeight, const double *forwardError, const int &forwardNodes, const double &learningRate) = 0;
	virtual void back_propagation_output(const double *prevOutput, const uint8_t *labels, const int &target, const double &learningRate) = 0;

	virtual void defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) = 0;
	virtual void deleteCuda() = 0;
	
	virtual double* getCudaOutputPointer() const = 0;
	virtual double* getCudaWeightPointer() const = 0;
	virtual double* getCudaErrorPointer() const = 0;

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
