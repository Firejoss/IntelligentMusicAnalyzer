#pragma once

#include "neuralDuino.h"

class NeuralNetwork
{

protected:
	vector<Neuron*> inputNodes;
	vector<vector<Neuron*>*> intermediateLayers;
	vector<Neuron*> outputNodes;
	Neuron* biasNode;

public:
	NeuralNetwork();
	NeuralNetwork(int inputVectorSize, vector<int> intermediateLayersSizes, int ouputVectorSize);
	~NeuralNetwork();

	int init(int inputVectorSize, vector<int> intermediateLayersSizes, int ouputVectorSize);

	float train(vector<TrainingSet> &trainingData_, float idealError, u_int maxEpochs);

	int feedInputs(TrainingSet &trainingInputValues);

	int propagate();

	float feedOutputIdealValues(TrainingSet & trainingSet);

	int backPropagate();

	int adjustWeights();

	void printOutput();

};

