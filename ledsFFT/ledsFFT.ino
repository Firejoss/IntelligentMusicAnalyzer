/*
 Name:		ledsFFT.ino
 Created:	12/02/2019 21:51:28
 Author:	joss
*/

#include <Audio.h>
#include <Easing.h>
#include "NeuralNetwork.h"

// --- GENERAL DEFINES ---
#define ON			1
#define OFF			0
#define G_SWITCH	ON
#define R_PIN		6
#define G_PIN		7
#define B_PIN		8 
#define MIC_PIN		PIN_A0 
#define SERIAL_DEBUG

//#define FFT1024 true // if commented out, FFT 256 resolution will be set by default

#ifdef FFT1024
#undef FFT256
#else
#define FFT256 true
#endif


/* ----- Neural network defines & variables ----- */

// recommended number of NN inputs given the data vector provided by the fft
#define FFT256_NN_NUM_INPUTS	128
#define FFT1024_NN_NUM_INPUTS	512

#ifdef FFT256
#define NUM_INPUTS		FFT256_NN_NUM_INPUTS
#elif FFT1024
#define NUM_INPUTS		FFT1024_NN_NUM_INPUTS
#endif
#define NUM_LAY_1		64
#define NUM_LAY_2		16
#define NUM_OUTPUTS		2

NeuralNetwork* bongoNeuralNetwork;
vector<TrainingSet> trainingData = {};
/* -----------------------------------------*/

// defines which bongo is to be played 
// for NN training and expected output vector
#define BONGO_NOISE_RECORDING	{0, 0}
#define BONGO_1_RECORDING		{0, 1}
#define BONGO_2_RECORDING		{1, 0}
vector<float> currentBongoRecording = BONGO_NOISE_RECORDING;

#define FFT_SUM_TRAINING_THRES  2.5

//-----------------------------------------------------------------------------------------------

// --- FFT display ---
AudioInputAnalog         adc1(14);

#ifdef FFT1024
AudioAnalyzeFFT1024      fft1;
#else
AudioAnalyzeFFT256		 fft1;
#endif

AudioConnection          patchCord0(adc1, fft1);


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

void pulseLed(int led_pin, float power) {

	if (power > 100) power = 100;
	if (power < 0) power = 0;
	else if (power < 0) power = 0;

	int dur = 200 * (0.35 + (power / 100));
	int max = 255 * (0.05 + power / 100);

	for (int pos = 0; pos < dur; pos++) {
		analogWrite(led_pin, max - Easing::easeOutExpo(pos, 0, max, dur));
		delay(2);
	}
}

void addTrainingSet(AudioStream* audioStream, vector<float> desiredOutput) {

	AudioAnalyzeFFT256* fft = nullptr;
#ifdef FFT256
	AudioAnalyzeFFT256* fft = static_cast<AudioAnalyzeFFT256*>(audioStream);
#elif FFT1024
	AudioAnalyzeFFT1024* fft = static_cast<AudioAnalyzeFFT1024*>(audioStream);
#endif
	
	if (fft == nullptr) return;

	if (fft->available() && fft->read(0, NUM_INPUTS/2 - 1) > FFT_SUM_TRAINING_THRES) {
		
		float fftSpectrum[NUM_INPUTS];
		
		for (int i = 0; i < NUM_INPUTS/2; i++) {
			fftSpectrum[i] = fft->read(i);
		}

		trainingData.push_back(TrainingSet(fftSpectrum, currentBongoRecording));
#ifdef DEBUG
		Serial.println("--- New training set added ! ---");
#endif
	}
}

//-------------------------------------------------------------------------------------------------
// ISR display handler

static IntervalTimer itimer;
//Initialise and start ISR timer
void initTimer(void) {
	//itimer.begin(refreshDisplay, 1000000.0 / ledUpdate_frequency);
}

//----------------------------------------------------------------------------------------------

float sigmoidFn(float in, byte isDerivative) {
	return isDerivative == HIGH ? sigmoidDerivative(in) : sigmoid(in);
}
float linear(float in, byte isDerivative) {
	return isDerivative == HIGH ? 1 : in;
}

void initDisplay() {
	pinMode(R_PIN, OUTPUT);
	pinMode(G_PIN, OUTPUT);
	pinMode(B_PIN, OUTPUT);
	digitalWrite(R_PIN, OFF);
	digitalWrite(G_PIN, OFF);
	digitalWrite(B_PIN, OFF);
}


void initNN() {
	bongoNeuralNetwork = new NeuralNetwork();
}


// ----------------------------------------------------
// ------------------- LOOP & SETUP -------------------
// ----------------------------------------------------

void setup() {
	AudioMemory(25);
	initDisplay();

	// creates the neural network
	initNN();

#ifdef SERIAL_DEBUG
	delay(300);
	Serial.begin(115200);
#endif

}

void loop() {

	// adding a training set to the training database if fft threshold exceeded
	addTrainingSet(static_cast<AudioStream*>(&fft1), currentBongoRecording);

	bongoNeuralNetwork->train(&trainingData);




	//if (!G_SWITCH) {
	//	return;
	//}

	// *** LEDS TEST ANIMATION ***

	//int val = analogRead(MIC_PIN);
	//Serial.println(val);

	//if (val > 480 && val < 550) {
	//	pulseLed(G_PIN, 0);
	//}
	//else if (val >= 550 && val < 600) {
	//	pulseLed(B_PIN, 80);
	//}
	//else if (val >= 600) {
	//	pulseLed(R_PIN, 50);
	//}
}