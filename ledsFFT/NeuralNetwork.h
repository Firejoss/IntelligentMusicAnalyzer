#pragma once
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "Arduino.h"

using namespace std;

//#define DEBUG
#define LEARNING_RATE 0.4
#define FALSE LOW
#define TRUE HIGH

#ifdef __arm__
// should use uinstd.h to define sbrk but Due causes a conflict
extern "C" char* sbrk(int incr);
#else  // __ARM__
extern char *__brkval;
#endif  // __arm__

#define sigmoid(x)			1.0 / (1.0 + (float)exp(-(float)(x)))
#define sigmoidPrime(x)		(float)(sigmoid(x) * (1.0 - sigmoid(x)))


typedef float(*activFn)(float, int);

struct Util {

	static vector<vector<float>> transpose(vector<vector<float>> const& v1) {

		if (v1.empty()) {
			return v1;
		}

		vector<vector<float>> v2;
		v2.resize(v1[0].size());

		for (auto & columnVect : v2) {
			columnVect.resize(v1.size());
		}

		for (size_t i = 0; i < v1.size(); i++)
		{
			for (size_t j = 0; j < v1[i].size(); j++)
			{
				v2[j][i] = v1[i][j];
			}
		}
		return v2;
	}

	// computes the dot product of two vectors
	static float dot(vector<float> const& v1, vector<float> const& v2) {

		if (v1.size() != v2.size()) {
			printMsgInts("Dot product error. Unequal vectors sizes : ", { v1.size(), v2.size() });
			return 0;
		}

		float result = 0;
		for (int i = 0, size = v1.size(); i < size; i++) {
			result += v1[i] * v2[i];
		}
		return result;
	}

	static void printMsg(String msg) {
		Serial.println(msg);
	}

	static void printMsgFloat(String msg, float argument) {
		Serial.print(msg);
		Serial.println(argument, 10);
	}	
	
	static void printMsgInt(String msg, int argument) {
		Serial.print(msg);
		Serial.println(argument);
	}	
	
	static void printMsgInts(String msg, vector<int> arguments) {
		Serial.print(msg);
		for (auto argument : arguments) {
			Serial.print(argument);
			Serial.print(", ");
		}
		Serial.print("\n");
	}		
};

struct TrainingSet {

	vector<float> inputValues;
	vector<float> idealOutputValues;

	TrainingSet()
	{}

	TrainingSet(float inputs_[], vector<float> desiredOutput_) :
		idealOutputValues(desiredOutput_)
	{
		inputValues.insert(inputValues.begin(), inputs_, inputs_ + sizeof(inputs_) / sizeof(inputs_[0]));
	}	
	
	TrainingSet(vector<float> inputs_, vector<float> desiredOutput_) :
		inputValues(inputs_),
		idealOutputValues(desiredOutput_)
	{}
};

struct Memory {

	static int getFreeMemory() {
		char top;
#ifdef __arm__
		return &top - reinterpret_cast<char*>(sbrk(0));
#elif defined(CORE_TEENSY) || (ARDUINO > 103 && ARDUINO != 151)
		return &top - __brkval;
#else  // __arm__
		return __brkval ? &top - __brkval : &top - __malloc_heap_start;
#endif  // __arm__
	}
};


class NeuralNetwork
{

protected:

	vector<vector<float>>			neuronOutputs;
	vector<vector<float>>			zs;			// neuron's inputs * weights + bias

	vector<vector<vector<float>>>	weights;
	vector<vector<float>>			biases;

	vector<vector<float>>			deltas;
	vector<float>					errors;		// computed substracting ideal values from outputs

public:
	NeuralNetwork();
	NeuralNetwork(int inputVectorSize, vector<int> intermediateLayersSizes, int ouputVectorSize);
	~NeuralNetwork();

	int init(vector<int> layersSizes_);

	int randomizeWeights();

	int randomizeBiases();

	float train(vector<TrainingSet> &trainingData_, float idealError, u_int maxEpochs);

	int feedInputs(TrainingSet &trainingInputValues);

	vector<float>* propagate();

	float feedOutputIdealValues(TrainingSet & trainingSet);

	int backpropagate();

	void printOutput();

};

#endif // NEURAL_NETWORK_H
