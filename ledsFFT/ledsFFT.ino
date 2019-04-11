/*
 Name:		ledsFFT.ino
 Created:	12/02/2019 21:51:28
 Author:	joss
*/

#include <Audio.h>
#include <Easing.h>
#include "NeuralNetwork.h"

namespace std {
	void __throw_bad_alloc()
	{
		Serial.println("Unable to allocate memory");
	}

	void __throw_length_error(char const*e)
	{
		Serial.print("Length Error :");
		Serial.println(e);
	}
}

// --- GENERAL DEFINES ---
#define ON						1
#define OFF						0
#define G_SWITCH				ON
#define R_PIN					6
#define G_PIN					7
#define B_PIN					8 
#define MIC_PIN					PIN_A3
#define BONGO_SELECT_BTN_PIN_1	15
#define BONGO_SELECT_BTN_PIN_2	20
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
#define NUM_LAY_2		32
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

#define FFT_SUM_TRAINING_THRES  1.5

//-----------------------------------------------------------------------------------------------

// --- FFT display ---
AudioInputAnalog         adc1(MIC_PIN);

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

void addTrainingSet(AudioStream* audioStream, vector<float> &idealOutput) {

#ifdef FFT256
	AudioAnalyzeFFT256* fft = static_cast<AudioAnalyzeFFT256*>(audioStream);
#elif FFT1024
	AudioAnalyzeFFT1024* fft = static_cast<AudioAnalyzeFFT1024*>(audioStream);
#endif
	
	if (fft == nullptr) return;

	if (fft->available() && fft->read(0, NUM_INPUTS/2 - 1) > FFT_SUM_TRAINING_THRES) {

#ifdef DEBUG
		Serial.println("--- Adding new FFT training set... ---");
#endif
		
		float fftSpectrum[NUM_INPUTS];
		
		for (int i = 0; i < NUM_INPUTS/2; i++) {
			fftSpectrum[i] = fft->read(i);
		}

		trainingData.push_back(TrainingSet(fftSpectrum, idealOutput));
#ifdef DEBUG
		Serial.print("--- New training set added ! --- idealOutput : ");
		Serial.print(idealOutput[0]);
		Serial.print(" - ");
		Serial.print(idealOutput[1]);
		Serial.print("--- Training data SIZE => ");
		Serial.println(trainingData.size());
#endif
	}
}

void testNeuralNetwork(NeuralNetwork* nn, AudioStream* audioStream) {

#ifdef FFT256
	AudioAnalyzeFFT256* fft = static_cast<AudioAnalyzeFFT256*>(audioStream);
#elif FFT1024
	AudioAnalyzeFFT1024* fft = static_cast<AudioAnalyzeFFT1024*>(audioStream);
#endif

	if (fft == nullptr) return;

	if (fft->available() && fft->read(0, NUM_INPUTS / 2 - 1) > FFT_SUM_TRAINING_THRES) {

#ifdef DEBUG
		Serial.println("--- Testing NN with real input... ---");
#endif

		float fftSpectrum[NUM_INPUTS];

		for (int i = 0; i < NUM_INPUTS / 2; i++) {
			fftSpectrum[i] = fft->read(i);
		}

		TrainingSet realInputData(fftSpectrum, {});

		nn->feedInputs(realInputData);
		nn->propagate();
#ifdef DEBUG
		nn->printOutput();
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

void initBongoRecordingSelectButtons() {
	//  --- RECORDED BONGO SELECTION BUTTONS ---
	/* 
	so that the NN knows which bongo (left or right or none) 
	it should ideally learn to recognize
	*/
	pinMode(21, OUTPUT);
	pinMode(BONGO_SELECT_BTN_PIN_2, INPUT);
	pinMode(19, OUTPUT);
	digitalWrite(21, LOW);
	digitalWrite(19, HIGH);

	pinMode(16, OUTPUT);
	pinMode(BONGO_SELECT_BTN_PIN_1, INPUT);
	pinMode(14, OUTPUT);
	digitalWrite(16, LOW);
	digitalWrite(14, HIGH);
}

void initDisplay() {
	pinMode(R_PIN, OUTPUT);
	pinMode(G_PIN, OUTPUT);
	pinMode(B_PIN, OUTPUT);
	digitalWrite(R_PIN, OFF);
	digitalWrite(G_PIN, OFF);
	digitalWrite(B_PIN, OFF);
}

void updateCurrentBongoRecordingSelection(vector<float> &currentBongosSelection) {
	
	currentBongosSelection = { digitalRead(BONGO_SELECT_BTN_PIN_1), digitalRead(BONGO_SELECT_BTN_PIN_2) };

//#ifdef DEBUG
//	Serial.print("BONGO RECORD SELECTION : ");
//	Serial.print(currentBongoRecording[0]);
//	Serial.print(" - ");
//	Serial.println(currentBongoRecording[1]);
//#endif

}


// ----------------------------------------------------
// ------------------- LOOP & SETUP -------------------
// ----------------------------------------------------

void setup() {

#ifdef SERIAL_DEBUG
	Serial.begin(115200);
#endif

	AudioMemory(25);
	initDisplay();
	initBongoRecordingSelectButtons();

	Serial.print("Free SRAM BEFORE NN => ");
	Serial.println(Memory::getFreeMemory());

	// creates the neural network
	bongoNeuralNetwork = new NeuralNetwork(NUM_INPUTS, { NUM_LAY_1, NUM_LAY_2 }, NUM_OUTPUTS);

	Serial.print("Free SRAM AFTER NN => ");
	Serial.println(Memory::getFreeMemory());
}

void loop() {

	while (trainingData.size() < 30) {
		// adding a training set to the training database if fft threshold exceeded
		updateCurrentBongoRecordingSelection(currentBongoRecording);
		addTrainingSet(&fft1, currentBongoRecording);
	}

	bongoNeuralNetwork->train(trainingData, 0.1, 1000);

	while (1) {

		testNeuralNetwork(bongoNeuralNetwork, &fft1);

		delay(500);
	}


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