#pragma once

#include <vector>
#include "Arduino.h"
using namespace std;

#define DEBUG 1
#define DISPLAY_ERROR 0
#define LEARNING_RATE 0.4
#define MOMENTUM 0.1
#define SPEED LEARNING_RATE
#define FALSE LOW
#define TRUE HIGH

#ifdef __arm__
// should use uinstd.h to define sbrk but Due causes a conflict
extern "C" char* sbrk(int incr);
#else  // __ARM__
extern char *__brkval;
#endif  // __arm__

#define sigmoid(x)			1.0 / (1.0 + (float)exp(-(float)(x)))
#define sigmoidPrime(x)		(float)( x * (1.0 - x) )


typedef float(*activFn)(float, int);

struct Util {

	// computes the dot product of two vectors
	static float dot(vector<float>& v1, vector<float>& v2) {

		if (v1.size() != v2.size()) {
			printMsg("Vectors size are not equal, dot product cannot be computed.");
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

