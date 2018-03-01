#pragma once

typedef enum LayerType { CONVOLUTIONAL, BATCH, FULLY_CONNECTED } LayerType;
typedef enum ActFctType { SIGMOID, TANH, RELU, NONE } ActFctType;

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

	virtual int getNodeCount(void) const = 0;
	virtual int getWeightCount(void) const = 0;
	virtual std::vector<double> getWeights(void) = 0;
	virtual std::vector<double> getBias(void) = 0;
	virtual int getPredictionIndex(void) = 0;

	virtual void forward_propagation(const double *prevOutput) = 0;

	//virtual void calcError(double *prevError, const int &prevNodes) = 0;
	virtual void back_propagation_output(const double *prevOutput, const uint8_t *labels, const int &target, const double &learningRate) = 0;
	virtual void back_propagation(const double *prevOutput, double *prevErr, const double &learningRate, const bool notFirst) = 0;

	virtual void defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) = 0;
	virtual void deleteCuda(void) = 0;

	virtual double* getCudaOutputPointer(void) const = 0;
	virtual double* getCudaWeightPointer(void) const = 0;
	//virtual double* getCudaErrorPointer(void) const = 0;
	virtual double* getCudaPrevErrorPointer(void) const = 0;

	inline LayerType getLayerType(void) const { return _l; }
	inline int getWidth(void) const { return _width; }
	inline int getHeight(void) const { return _height; }
	inline int getDepth(void) const { return _depth; }
	inline ActFctType getActivationFunction(void) const { return _a; }

	virtual void printW() = 0;

protected:
	LayerType _l;
	ActFctType _a;
	int _width;
	int _height;
	int _depth;
};
