#include "neuralDuino.h"

Neuron::Neuron() {
	beta = 0;
	output = 0;
	inCount = 0;
	input = NULL;
	synWeight = NULL;
	prevDelWeight = NULL;
}

void Neuron::begin(int num_syn, int noConnections = FALSE, int noInputs = FALSE) {
	//deallocating previously allocated memory
	delete input;
	vector<Neuron*>().swap(inNodes);
	delete synWeight;
	delete prevDelWeight;
	numSynapse = num_syn;
	if (num_syn == 0) {
		// since no memory is requested just return
		return;
	}
	//allocating memory only if requested
	if (noInputs == FALSE) { input = new float[num_syn]; }
	if (noConnections == FALSE) { inNodes.resize(num_syn); }
	if (noConnections == FALSE || noInputs == FALSE) {
		synWeight = new float[num_syn];
		prevDelWeight = new float[num_syn];
	}

	// ANALOGREAD not usable when FFT is used... find another seed method later
	randomSeed(45452/*analogRead(A0)*/);
	for (int i = 0; i < num_syn; i++) {
		synWeight[i] = (float)(((float)random(0, 100) / (float)100) - 0.2);
		prevDelWeight[i] = 0; //important to initialize allocated memory
	}
}

void Neuron::setInput(float inputVals[]) {
	float sum = 0;
	inCount = 0; //make sure that inCount is marked as zero for inputNodes

	for (int i = 0; i < numSynapse; i++) {
		sum = sum + (synWeight[i] * inputVals[i]);
		input[i] = inputVals[i]; //copying by value
	}
	output = activation(sum, LOW);
}

void Neuron::setInput(int inputVals[]) {
	float sum = 0;
	inCount = 0; //make sure that inCount is marked as zero for inputNodes

	for (int i = 0; i < numSynapse; i++) {
		sum = sum + (synWeight[i] * inputVals[i]);
		input[i] = inputVals[i]; //copying by value
	}
	output = activation(sum, LOW);
}

void Neuron::setOutput(float value) {
	output = value;
	inCount = 0;	//only to be used for non input nodes like bias
}

float Neuron::propagate() {
	//this function is called once on the last layer neuron/neurons
	//therefore the output for each of these is stored inside these neurons
	//itself for future adjustment of weights
	//Serial.print("inCount is  ");
	//Serial.println(inCount);
	//Serial.flush();
	float sum = 0;
	if (inCount > 0) {
		int temp = inCount;
		while (temp != 0) {
			temp--;
			sum = sum + (synWeight[temp] * inNodes[temp]->propagate());
		}
		output = activation(sum, LOW);
	}

	//		Serial.print((int)this);
		//	Serial.print("->");
			//Serial.println(output);
	return  output;

}


float Neuron::setIdealOutput(float desiredOutput) {
	beta = desiredOutput - output;
#if DISPLAY_ERROR
	Serial.println((int)(beta * 100));
#endif
	return ((int)(beta * 100) == 0 ? 1 : 0);
}

/*
this function is called on all those nodes that have an input node
*/
void Neuron::backpropagate() {
	float myDelta = beta * activation(output, HIGH);
	if (inCount != 0) {
		int temp = inCount;
		while (temp--) {
			//back propagating the delta to previous layer
			inNodes[temp]->beta = inNodes[temp]->beta + (synWeight[temp] * myDelta);
			//by this all the betas reach the previous layer nodes as summed up
		}
	}

}
/*
this is called on every node after complete backpropagation is done for all nodes
*/
void Neuron::adjWeights() {
	float myDelta = beta * activation(output, HIGH);
	if (inCount != 0) { // inNodes is filled up 
		int temp = inCount;
		while (temp != 0) {
			temp--;
			float delWeight = (SPEED * inNodes[temp]->output * myDelta);
			synWeight[temp] = synWeight[temp] + delWeight + MOMENTUM * prevDelWeight[temp];
			prevDelWeight[temp] = delWeight;
			//Serial.println(prevDelWeight[temp]);
			//Serial.flush();
		}
	}
	else { // inNodes is empty, therfore this is input node
		for (int i = 0; i < numSynapse; i++) {
			float  delWeight = (SPEED * input[i] * myDelta);
			synWeight[i] = synWeight[i] + delWeight + MOMENTUM * prevDelWeight[i];
			prevDelWeight[i] = delWeight;
			//Serial.println(prevDelWeight[i]);
			//Serial.flush();
		}
	}
#if DEBUG
	Serial.print((int)this);
	Serial.print("=this,beta=");
	Serial.print(beta);
	Serial.print(",out=");
	Serial.println(output);
	Serial.flush();
#endif
	beta = 0;
}

void Neuron::printWeights() {
	for (int i = 0; i < numSynapse; i++) {
		Serial.print(synWeight[i]);
		Serial.print(",");
	}
	Serial.println();

}

void Neuron::connectInput(Neuron* inNode) {

	inNodes.push_back(inNode);
	inCount = inNodes.size();
#if DEBUG
	Serial.print((int)this);
	Serial.print(F(" : connected to :"));
	Serial.println((int)inNode);
#endif
	//Serial.println((int)inNodes[inCount-1]);
}

void Neuron::connectInputs(vector<Neuron*> &inNodes_) {
	inNodes.insert(inNodes.end(), inNodes_.begin(), inNodes_.end());
	inCount = inNodes.size();
}

void Neuron::setActivationFn(activFn userFn) {
	this->activation = userFn;

#if DEBUG
	Serial.print(F("ActFN is "));
	Serial.println((int)userFn);
#endif
}