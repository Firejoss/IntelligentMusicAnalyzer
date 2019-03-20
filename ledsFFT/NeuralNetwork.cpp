
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::NeuralNetwork(int inputVectorSize_, vector<int> intermediateLayersSizes_, int ouputVectorSize_)
{
	init(inputVectorSize_, intermediateLayersSizes_, ouputVectorSize_);
}

NeuralNetwork::~NeuralNetwork()
{
}

int NeuralNetwork::init(int inputVectorSize_, vector<int> intermediateLayersSizes_, int ouputVectorSize_) {

	// --- BIAS node ---
	biasNode = new Neuron();
	biasNode->begin(0);
	biasNode->setActivationFn(&Neuron::sigmoidFn);
	biasNode->setOutput(1);

	Serial.println("biasNode initialized");

	// --- INPUTS ---
	inputNodes.resize(inputVectorSize_);

	for (auto &inputNode : inputNodes) {
		inputNode = new Neuron();
		inputNode->begin(0);
		inputNode->setActivationFn(&Neuron::linear);
	}
	Serial.println("inputNodes initialized");

	// --- INTERMEDIATE LAYERS ---
	for (int i; i < intermediateLayersSizes_.size(); i++) {
		
		vector<Neuron*> intermLayer(intermediateLayersSizes_[i], new Neuron());
		intermediateLayers.push_back(intermLayer);
		Serial.print("nodeslayer created, size: ");
		Serial.println(intermediateLayers[i].size());

		for (int j = 0; j < intermediateLayers[i].size(); j++) {
			intermediateLayers[i][j]->begin(0 == i ? inputNodes.size() : intermediateLayers[i-1].size());
			intermediateLayers[i][j]->setActivationFn(&Neuron::sigmoidFn);
			intermediateLayers[i][j]->connectInputs(inputNodes);
			intermediateLayers[i][j]->connectInput(biasNode);
		}

	}
	Serial.println("layerNodes initialized");

	// --- OUTPUT NODES ---
	outputNodes.resize(ouputVectorSize_);

	for (auto &outputNode : outputNodes) {
		outputNode = new Neuron();
		outputNode->begin(intermediateLayers.back().size());
		outputNode->setActivationFn(&Neuron::sigmoidFn);
		outputNode->connectInputs(intermediateLayers.back());
		outputNode->connectInput(biasNode);
	}
	Serial.println("outputNodes initialized");

	return 0;
}

float NeuralNetwork::train(vector<TrainingSet> &trainingData_, float idealError, u_int maxEpochs) {

	float error = idealError + 1;
	u_int numEpochs = 0;

	while (error > idealError || numEpochs < maxEpochs) {

		numEpochs++;

		for (auto &trainingSet : trainingData_) {

			feedInputs(trainingSet);
			propagate();
			error = feedOutputIdealValues(trainingSet);
			backPropagate();
			adjustWeights();
		}

	}

	return error;
}

int NeuralNetwork::feedInputs(TrainingSet &trainingSet) {

	if (trainingSet.inputValues.size() != this->inputNodes.size()) {
		return -1;
	}
	int inputSize = this->inputNodes.size();
	for (int i = 0; i < inputSize; i++) {
		this->inputNodes[i]->setOutput(trainingSet.inputValues[i]);
	}
	return 0;
}


int NeuralNetwork::propagate() {

	if (outputNodes.empty()) {
		return -1;
	}	
	for (auto &outputNode : outputNodes) {
		outputNode->propagate();
	}	
	return 0;
}

int NeuralNetwork::feedOutputIdealValues(TrainingSet &trainingSet) {
	
	if (trainingSet.idealOutputValues.size() != this->outputNodes.size()) {
		return -1;
	}
	int outputSize = this->outputNodes.size();
	for (int i = 0; i < outputSize; i++) {
		this->outputNodes[i]->setIdealOutput(trainingSet.idealOutputValues[i]);
	}
	return 0;
}

int NeuralNetwork::backPropagate() {

	if (outputNodes.empty()) {
		return -1;
	}	
	for (auto &outputNode : outputNodes) {
		outputNode->propagate();
	}	
	return 0;
}


int NeuralNetwork::adjustWeights() {

	for (auto &outputNode : this->outputNodes) {
		outputNode->adjWeights();
	}

	vector<vector<Neuron*>>::reverse_iterator rit = this->intermediateLayers.rbegin();
	
	for (; rit != this->intermediateLayers.rend(); ++rit) {
		
		for (auto &intermediateNode : *rit) {

			intermediateNode->adjWeights();
		}
	}
	return 0;
}
