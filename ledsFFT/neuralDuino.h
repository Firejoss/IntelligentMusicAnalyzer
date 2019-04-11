#ifndef __neuralDuino_h__
#define __neuralDuino_h__

#define DEBUG 0
#define DISPLAY_ERROR 0
#define LEARNING_RATE 0.4
#define MOMENTUM 0.1
#define SPEED LEARNING_RATE
#define FALSE LOW
#define TRUE HIGH

#include "Arduino.h"
#define sigmoid(x)           (1.0 / (1.0 + (float)exp(-(float)(x))))
#define sigmoidDerivative(x) ((float)((x)*(1.0-(x))))

#include <vector>
using namespace std;

#ifdef __arm__
// should use uinstd.h to define sbrk but Due causes a conflict
extern "C" char* sbrk(int incr);
#else  // __ARM__
extern char *__brkval;
#endif  // __arm__

/*
	types of neurons can be classified using the 2 begin() parameters noConnections and noInputs

									noConnections	noInputs
	hybrid (most inefficient)		LOW				LOW			(DEFAULT) //leaves the control to the programmer
	intermediate neuron				LOW				HIGH
	bias							HIGH			HIGH
	input neuron					HIGH			LOW

*/

/*
	TODO:
	-too many float calculations consider optimizing /removing these as well
*/

typedef float(*activFn)(float, int);


struct TrainingSet {

	vector<float> inputValues;
	vector<float> idealOutputValues;

	TrainingSet() 
	{}

	TrainingSet(float* inputs_, vector<float> desiredOutput_) :
		inputValues(vector<float>(inputs_, inputs_ + sizeof(inputs_) / sizeof(inputs_[0]))),
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


class Neuron {

public:

	static float sigmoidFn(float in, int isDerivative) {
		return isDerivative == HIGH ? sigmoidDerivative(in) : sigmoid(in);
	}
	static float linear(float in, int isDerivative) {
		return isDerivative == HIGH ? 1 : in;
	}

	//constructor
	Neuron();

	/*
	synWeight holds the weight of each synapase that is INCOMING for that neuron
	the length of synWeight is decided by the begin() function where mem alloc
	takes place.
	*/
	vector<float> synWeights;

	/*
	the second way to have inputs is to have output of some other neurons feed into
	this neuron inNodes stores the address of every neuron which is connected to this
	neuron and asks for the output from each of those to calculate its output
	*/
	vector<Neuron*>* inNodes;	// pointer to on array of pointers of type neuron

	float bias;
	
	/*
	stores the output of this neuron
	*/
	float output;

	/*
	keeps count of the inputs specified by setInput() and begin()
	this counts the number of float array type inputs and not the connectInput() ones
	*/
	int numSynapse;

	/*
	associates an activation function for this neuron, user sets the address of any activation
	function using setActivationFn()
	*/
	activFn activation;

	float beta; //just another backpropagation constant

	/*
	to prevent memory fragmentation while retaining flexibility the begin function
	allocates memory one time only, while it is possible to do away with this
	and have completely dynamic allocn and reallocn but it is not advisable because the RAM may get
	fragmented

	ARGs:
	noConnections : implies that this neuron will not have any connections coming into it
					when noConnections = HIGH, this doesnt allocate memory for the same
	noInputs	  : implies that this neuron will not have any float inputs coming into it from the user
					when noInputs = HIGH, this doesnt allocate memory for the same
	these are optional arguments for those people who know what they are doing
	*/
	void begin(int num_syn, int noConnections = FALSE, int noInputs = FALSE);
	/*
	adjust weights according to the update rule
	*/
	void adjWeights();
	/*
	backpropagate the betas to the connected nodes only
	*/
	void backpropagate();
	/*
	setDesiredOUtput only valid for the last nodes
	*/
	float setIdealOutput(float desiredOutput);

	/*
	print the final weights after learning has happened
	*/
	void printWeights();

	/*
	connect an array of pointers to nodes from which
	ouputs will be taken as inputs to this->node
	*/
	void connectInputs(vector<Neuron*>* inNodes_);
	/*
	compute output value from the input nodes, store it and return it
	*/
	float propagate();
	/*
	set the activation functionfor this->node
	*/
	void setActivationFn(activFn userFn);

	static void printMessage(String msg, float argument) {
		Serial.print(msg);
		Serial.println(argument, 10);
	}
};

#endif
