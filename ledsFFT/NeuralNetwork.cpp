
#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::NeuralNetwork(int inputVectorSize_, vector<int> intermediateLayersSizes_, int ouputVectorSize_)
{
	intermediateLayersSizes_.insert(intermediateLayersSizes_.begin(), inputVectorSize_);
	intermediateLayersSizes_.push_back(ouputVectorSize_);
	init(intermediateLayersSizes_);
}

NeuralNetwork::~NeuralNetwork()
{
}

int NeuralNetwork::init(vector<int> layersSizes_) {

	// --- INPUT, INTERMEDIATE AND OUTPUT LAYERS ---

	weights.resize(layersSizes_.size() - 1);
	biases.resize(layersSizes_.size() - 1);
	zs.resize(layersSizes_.size() - 1);

	// error vector has the same output vector's size
	errors.resize(layersSizes_.back());

	deltas.resize(layersSizes_.size() - 1);
	
	neuronOutputs.resize(layersSizes_.size());
	neuronOutputs.front().resize(layersSizes_.front());

	for (int i = 0; i < layersSizes_.size(); i++) {
		neuronOutputs[i].resize(layersSizes_[i]);
	}

	for (int j = 1; j < layersSizes_.size(); j++) {
		
		zs[j - 1].resize(layersSizes_[j]);
		biases[j - 1].resize(layersSizes_[j]);
		weights[j - 1].resize(layersSizes_[j]);

		for (int k = 0; k < layersSizes_[j]; k++) {
			weights[j - 1][k].resize(layersSizes_[j - 1]);
		}
	}

	randomizeWeights();
	randomizeBiases();

	Util::printMsg("Neural network initialized !");

	return 0;
}

int NeuralNetwork::randomizeWeights() {

	// weights ordered by layers of neurons
	for (auto& layer : weights) {

		// sublayers containing the weights of the connections
		// between one neuron and each neuron of the next layer
		for (auto& subLayer : layer) {

			// every weight initialized randomly in [-0.99 ; 0.99]

			for (auto& weight : subLayer) {

				weight = (float)(rand() % 200 - 100) / 100.0;

			}
		}
	}
	return 0;
}

int NeuralNetwork::randomizeBiases() {

	// biases ordered by layer of neurons
	for (auto& layer : biases) {

		// every weight initialized randomly in [-0.99 ; 0.99]
		for (auto& bias : layer) {

			bias = (float)(rand() % 200 - 100) / 100.0;

		}
	}
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
		Util::printMsgInt("\nEpoch ", numEpochs);

		for (auto &trainingSet : trainingData_) {

			Serial.println("Feed inputs...");
			feedInputs(trainingSet);
			Serial.println("Propagate...");
			propagate();

			Serial.println("Process error...");
			error = feedOutputIdealValues(trainingSet);
			Util::printMsgFloat("Error => ", error);

			backpropagate();
		}

	}

	return error;
}

int NeuralNetwork::feedInputs(TrainingSet &trainingSet) {

	if (trainingSet.inputValues.size() != neuronOutputs.front().size()) {
		Util::printMsgInts("input and trainingset vectors have different sizes : ", 
			{ trainingSet.inputValues.size(), neuronOutputs.front().size() });
		return -1;
	}
	int inputSize = neuronOutputs.front().size();
	for (int i = 0; i < inputSize; i++) {
		neuronOutputs.front()[i] = trainingSet.inputValues[i];
	}

#ifdef DEBUG
	Serial.print("\n--- input values : ---\n\n--> ");
	for (auto &inputVal : neuronOutputs.front()) {
		Serial.print(inputVal);
		Serial.print(" | ");
	}
	Serial.print("\n\n");

#endif // DEBUG

	return 0;
}


vector<float>* NeuralNetwork::propagate() {

	int layersNum = neuronOutputs.size();

	// begin at 1 because 0 is the input layer;
	for (int i = 1; i < layersNum; i++) {

		for (int j = 0; j < neuronOutputs[i].size(); j++) {

			float sum = 0;
			for (int k = 0; k < neuronOutputs[i - 1].size(); k++) {
				sum += neuronOutputs[i - 1][k] * weights[i - 1][j][k];
			}

			zs[i - 1][j] = sum + biases[i - 1][j];

			neuronOutputs[i][j] = sigmoid(zs[i - 1][j]);
		}
	}
	return &(neuronOutputs.back());
}

float NeuralNetwork::feedOutputIdealValues(TrainingSet &trainingSet) {

	if (trainingSet.idealOutputValues.size() != neuronOutputs.back().size()) {
		Serial.println("Ideal output and output vectors have different sizes");
		return -1;
	}

	int outputNum = neuronOutputs.back().size();
	float error = 0;

	for (int i = 0; i < outputNum; i++) {
		errors[i] = neuronOutputs.back()[i] - trainingSet.idealOutputValues[i];
		error += errors[i];
	}
	return error;
}

int NeuralNetwork::backpropagate() {

	//-- first initating backward pass --
	for (int i = 0; i < errors.size(); i++) {

		deltas.back().push_back(errors[i] * sigmoidPrime(zs.back()[i]));

		biases.back()[i] = deltas.back().back();

		for (int j = 0; j < weights[weights.size() - 2].size(); j++) {

			weights.back()[i][j] = deltas.back()[i] * neuronOutputs[neuronOutputs.size() - 2][j];
		}
	}
	// ----------------------------------

	for (int k = deltas.size() - 2; k >= 0; k--) {

		for (int l = 0; l < weights[k].size(); l++) {

			for (int m = 0; m < weights[k][l].size(); m++) {

				vector<float> transpWeights = {};
				for (int n = 0; n < weights[k + 1].size(); n++) {
					transpWeights.push_back(weights[k + 1][n][l]);
				}

				deltas[k][l] = Util::dot(deltas[k + 1], transpWeights) * sigmoidPrime(zs[k][l]);
			
				biases[k][l] = deltas[k][l];

				for (int p = 0; p < weights[k][l].size(); p++) {

					weights[k][l][p] = deltas[k][l] * neuronOutputs[k][p];
				}

			}
		}

	}
#ifdef DEBUG
	Util::printMsg("backpropagation done.");
#endif

	return 0;
}

void NeuralNetwork::printOutput() {

	Serial.print("\n--- Neural Network Output : | ");

	for (auto &outputVal : neuronOutputs.back()) {
		Serial.print(outputVal);
		Serial.print(" | ");
	}
	Serial.print("\n");
}
