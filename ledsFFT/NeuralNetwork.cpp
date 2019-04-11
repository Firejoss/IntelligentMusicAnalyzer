
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

	// --- INPUTS ---
	inputNodes.resize(inputVectorSize_);

	for (auto &inputNode : inputNodes) {
		inputNode = new Neuron();
		inputNode->begin(0);
		inputNode->setActivationFn(&Neuron::linear);
	}
	Serial.println("Input nodes initialized\n");

	// --- INTERMEDIATE LAYERS ---
	for (int i; i < intermediateLayersSizes_.size(); i++) {
		Serial.println(Memory::getFreeMemory());
		Neuron* interNeuron = nullptr;
		vector<Neuron*>* intermLayer = new vector<Neuron*>();
		for (int index = 0; index < intermediateLayersSizes_[i]; index++) {
			interNeuron = new Neuron();
			intermLayer->push_back(interNeuron);
		}

		intermediateLayers.push_back(intermLayer);
		Serial.print("Intermediate layer created, size: ");
		Serial.println(intermediateLayers.back()->size());

		for (int j = 0; j < intermediateLayers.back()->size(); j++) {
			(*(intermediateLayers[i])) [j]->begin(0 == i ? inputNodes.size() : intermediateLayers[i-1]->size());
			(*(intermediateLayers[i])) [j]->setActivationFn(&Neuron::sigmoidFn);
			(*(intermediateLayers[i])) [j]->connectInputs(0 == i ? &inputNodes : intermediateLayers[i - 1]);
		}

	}
	
	// --- OUTPUT NODES ---
	outputNodes.resize(ouputVectorSize_);

	for (auto &outputNode : outputNodes) {
		outputNode = new Neuron();
		outputNode->begin(intermediateLayers.back()->size());
		outputNode->setActivationFn(&Neuron::sigmoidFn);
		outputNode->connectInputs(intermediateLayers.back());
	}
	Serial.println("Output nodes initialized\n\n");

	return 0;
}

float NeuralNetwork::train(vector<TrainingSet> &trainingData_, float idealError, u_int maxEpochs) {

	float error = 1 + idealError;
	u_int numEpochs = 0;

#ifdef DEBUG
	Serial.println("\n--- Starting NN training ---");
#endif // DEBUG

	while (error > idealError && numEpochs < maxEpochs) {

		numEpochs++;
		Neuron::printMessage("Epoch ", numEpochs);

		for (auto &trainingSet : trainingData_) {

			Serial.println("Feed inputs...");
			feedInputs(trainingSet);
			Serial.println("Propagate...");
			propagate();

			Serial.println("Process error...");
			error = feedOutputIdealValues(trainingSet);
			Neuron::printMessage("Error => ", error);

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
		this->inputNodes[i]->output = trainingSet.inputValues[i];
	}

#ifdef DEBUG
	Serial.print("\n--- Printing Input values : ---\n\n--> ");
	for (auto node : inputNodes) {
		Serial.print(node->output);
		Serial.print(" | ");
	}
	Serial.print("\n\n");

#endif // DEBUG


	return 0;
}


int NeuralNetwork::propagate() {

	if (outputNodes.empty()) {
		return -1;
	}	
	for (int i = 0; i < outputNodes.size(); i++) {
		outputNodes[i]->propagate();
	}	
	return 0;
}

int NeuralNetwork::feedOutputIdealValues(TrainingSet &trainingSet) {
	
	if (trainingSet.idealOutputValues.size() != this->outputNodes.size()) {
		Serial.println("Ideal output and output vectors have different sizes");
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

	for (auto outputNode : this->outputNodes) {
		outputNode->backpropagate();
	}

	vector<vector<Neuron*>*>::reverse_iterator rit = intermediateLayers.rbegin();

	for (; rit != intermediateLayers.rend(); ++rit) {

		for (auto intermediateNode : **rit) {

			intermediateNode->backpropagate();
		}
	}
	return 0;
}

int NeuralNetwork::adjustWeights() {

	for (auto outputNode : this->outputNodes) {
		outputNode->adjWeights();
	}

	vector<vector<Neuron*>*>::reverse_iterator rit = intermediateLayers.rbegin();
	
	for (; rit != intermediateLayers.rend(); ++rit) {
		
		for (auto intermediateNode : **rit) {

			intermediateNode->adjWeights();
		}
	}
	return 0;
}

void NeuralNetwork::printOutput() {

	Serial.print("--- Neural Network Output : [ - ");

	for (auto &outputNode : this->outputNodes) {
		Serial.print(outputNode->output);
		Serial.print(" - ");
	}
	Serial.print("]\n");
}
