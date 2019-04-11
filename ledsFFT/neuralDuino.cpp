#include "neuralDuino.h"

Neuron::Neuron() {
	beta = 0;
	output = 0;
}

void Neuron::begin(int num_syn, int noConnections = FALSE, int noInputs = FALSE) {
	// deallocating previously allocated memory
	vector<float>().swap(synWeights);
	numSynapse = num_syn;
	if (num_syn == 0) {
		// since no memory is requested just return
		inNodes = nullptr;
		bias = 0;
		return;
	}
	
	if (noConnections == FALSE || noInputs == FALSE) {
		synWeights.resize(num_syn);
	}

	for (int i = 0; i < num_syn; i++) {
		synWeights[i] = (float)(rand() % 100) / 100.0;
	}
	bias = (float) (rand() % 100) / 100.0;
}

float Neuron::propagate() {
	// this function is called once on the last layer neuron/neurons
	// therefore the output for each of these is stored inside these neurons
	// itself for future adjustment of weights

	float sum = 0;
	if (inNodes != nullptr) {
		//printMessage("Neuron inputs number : ", inNodes->size());
		int temp = inNodes->size();
		while (temp != 0) {
			temp--;
			sum += synWeights[temp] * (*inNodes)[temp]->propagate();
		}
		output = activation(sum + bias, LOW);
	}

	return output;
}


float Neuron::setIdealOutput(float desiredOutput) {
	beta = output - desiredOutput;
#if DISPLAY_ERROR
	Serial.println(beta, 10);
#endif
	return beta;
}

/* this function is called on all nodes that have an input node */
void Neuron::backpropagate() {
	
	float delta = beta * activation(output, HIGH);

	int temp = inNodes->size();
	
	if (temp != 0) {

		while (temp--) {
			// back propagating the delta to previous layer
			(*inNodes)[temp]->beta += synWeights[temp] * delta;
			// by this all the betas reach the previous layer nodes as summed up
		}
	}

}

/* this is called on every node after complete backpropagation is done for all nodes */
void Neuron::adjWeights() {
	
	float delta = beta * activation(output, HIGH);

	bias += SPEED * delta;
	
	int temp = inNodes->size();
	if (temp > 0) {			// inNodes is filled up 
		while (temp != 0) {
			temp--;
			synWeights[temp] += SPEED * (*inNodes)[temp]->output * delta;
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
		Serial.print(synWeights[i]);
		Serial.print(",");
	}
	Serial.println();

}

void Neuron::connectInputs(vector<Neuron*>* inNodes_) {
	inNodes = inNodes_;
}

void Neuron::setActivationFn(activFn userFn) {
	this->activation = userFn;

#if DEBUG
	Serial.print(F("ActFN is "));
	Serial.println((int)userFn);
#endif
}