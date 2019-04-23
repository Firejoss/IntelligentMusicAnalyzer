/*
 Name:		ledsFFT.ino
 Created:	12/02/2019 21:51:28
 Author:	joss
*/

#include <map>
#include <Easing.h>
#include <Audio.h>
#include "config.h"
#include "NeuralNetwork.h"
#ifdef SAVE_TRAINING_DATA_SDCARD
	#include <SD.h>
	#include <SPI.h>
#endif

NeuralNetwork* bongoNeuralNetwork;
vector<TrainingSet> trainingData;

TrainingSet* realInputData = new TrainingSet(NN_INPUT_SIZE, NN_OUTPUT_SIZE);

vector<nn_double> currentBongoRecording = BONGO_NOISE_RECORDING;

//-----------------------------------------------------------------------------------------------

// --- FFT display ---
AudioInputAnalog         adc1(MIC_PIN);

#ifdef FFT1024
AudioAnalyzeFFT1024      fft1;
#else
AudioAnalyzeFFT256		 fft1;
#endif

AudioConnection          patchCord0(adc1, fft1);

vector<float>			lastFFTOutput;
vector<int>				lastFFTMaxPeaks;


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

void pulseLed(int led_pin, nn_double power) {

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

bool detectAttack() {

	int startIndex = 2;				// ignore 0 and 1 index for irrelevant fft values

	bool hasThresholdPeak = false;
	vector<float> fftOutput;
	vector<float> fftDeltas;

	for (size_t i = startIndex; i < FFT_INPUT_SIZE; i++) {

		float val = fft1.read(i);
		fftOutput.push_back(val > 0.02 ? val : 0);
		fftDeltas.push_back(fftOutput.back() - lastFFTOutput[i - startIndex]);

		hasThresholdPeak = hasThresholdPeak || val > FFT_THRESHOLD;
	}
	if (!hasThresholdPeak) {
		return false;
	}

	// check if new peaks appeared
	float deltasSum = 0;
	for (size_t j = 0; j < fftDeltas.size(); j++) {
		deltasSum += fftDeltas[j] > 0.02 ? fftDeltas[j] : 0;
	}
	if (deltasSum < 0.30) {
		return false;
	}

	// get max peaks indexes
	vector<int> fftMaxPeaks;
	vector<float> fftOutputTemp(fftOutput.begin(), fftOutput.end());

	while (fftMaxPeaks.size() < NN_INPUT_SIZE)
	{
		int outputTempCount = fftOutputTemp.size();
		float maxVal = 0;
		int maxValIndex = 0;

		for (size_t k = 0; k < outputTempCount; k++)
		{
			if (fftOutputTemp[k] > maxVal) {
				maxVal = fftOutputTemp[k];
				maxValIndex = k;
			}
		}
		fftMaxPeaks.push_back(maxValIndex);
		fftOutputTemp[maxValIndex] = 0;
	}

	// remove duplicate adjacent peaks
	fftMaxPeaks.erase(
		std::remove(fftMaxPeaks.begin(), fftMaxPeaks.end(), 0),
		fftMaxPeaks.end());

	for (size_t m = 0; m < fftMaxPeaks.size(); m++)
	{
		fftMaxPeaks.erase(
			std::remove(fftMaxPeaks.begin(), fftMaxPeaks.end(), fftMaxPeaks[m] + 1),
			fftMaxPeaks.end());
		fftMaxPeaks.erase(
			std::remove(fftMaxPeaks.begin(), fftMaxPeaks.end(), fftMaxPeaks[m] - 1),
			fftMaxPeaks.end());
		fftMaxPeaks.erase(
			std::remove(fftMaxPeaks.begin(), fftMaxPeaks.end(), fftMaxPeaks[m] + 2),
			fftMaxPeaks.end());
		fftMaxPeaks.erase(
			std::remove(fftMaxPeaks.begin(), fftMaxPeaks.end(), fftMaxPeaks[m] - 2),
			fftMaxPeaks.end());		
	}

	// checking of max peaks are not the same from previous fft
	int similarityCount = 0;
	for (size_t m = 0; m < fftMaxPeaks.size(); m++)
	{
		for (size_t n = 0; n < lastFFTMaxPeaks.size(); n++)
		{
			int deltaPeaks = abs(lastFFTMaxPeaks[n] - fftMaxPeaks[m]);
			if (deltaPeaks < 2) similarityCount++;
		}
	}
	if (similarityCount > 2) {
		Util::printMsg("--s--");
		return false;
	}

	vector<float> peakValues;
	for (auto& peakIndex : fftMaxPeaks) {
		peakValues.push_back(fftOutput[peakIndex]);
	}
	Util::printMsgInts("\nFFT Max Peak Indexes => ", fftMaxPeaks);
	Util::printMsgFloats("FFT Max Peak VALUES => ", peakValues);

	lastFFTOutput = fftOutput;
	lastFFTMaxPeaks = fftMaxPeaks;
	return true;

}

void addTrainingSet(vector<nn_double> &idealOutput) {
	
	if (!fft1.available())	return;
	if (!detectAttack())	return;

#ifdef DEBUG
	Serial.println("--- Adding new FFT training set... ---");
#endif
		
	int fftSize = lastFFTMaxPeaks.size();
	for (int i = 0; i < NN_INPUT_SIZE; i++) {
		//realInputData->inputValues[i] = lastFFTOutput[i];

		realInputData->inputValues[i] = lastFFTMaxPeaks[i];
	}

#ifdef SAVE_TRAINING_DATA_SDCARD
	saveTrainingSetSDCard(*realInputData);
#else
	realInputData->idealOutputValues = idealOutput;
	trainingData.push_back(*realInputData);
#endif

#ifdef DEBUG
	Util::printMsgFloats("--- New training set added --- idealOutput : ", { idealOutput[0], idealOutput[1] });
	Util::printMsgInt("--- Training data SIZE => ", trainingData.size());
#endif

#ifdef DEBUG_MEMORY
	Util::printMsgInts("Free SRAM AFTER adding TSet #", { trainingData.size(), Memory::getFreeMemory() });
#endif

}

void testNeuralNetwork(NeuralNetwork* nn) {

#ifdef DEBUG_MEMORY
	//Util::printMsgInt("Free SRAM before NN real testing : ", Memory::getFreeMemory());
#endif

	if (!fft1.available())	return;
	if (!detectAttack())	return;

#ifdef DEBUG
	Serial.println("*** Testing NN with real input ***");
#endif

	for (int i = 0; i < NN_INPUT_SIZE; i++) {
		realInputData->inputValues[i] = lastFFTMaxPeaks[i];
	}
	nn->feedInputs(*realInputData);
	nn->propagate();
	nn->printOutput();
}

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

void updateCurrentBongoRecordingSelection(vector<nn_double> &currentBongosSelection) {
	
	currentBongosSelection = { digitalRead(BONGO_SELECT_BTN_PIN_1), digitalRead(BONGO_SELECT_BTN_PIN_2) };

#ifdef DEBUG_BONGO_SELECTION
	Serial.print("BONGO RECORD SELECTION : ");
	Serial.print(currentBongoRecording[0]);
	Serial.print(" - ");
	Serial.println(currentBongoRecording[1]);
#endif

}

// ----------------------------------------------------
// ------------------- LOOP & SETUP -------------------
// ----------------------------------------------------

void setup() {

#ifdef DEBUG_SERIAL
	Serial.begin(115200);
#endif

	AudioMemory(10);
	initDisplay();
	initBongoRecordingSelectButtons();

#ifdef DEBUG_MEMORY
	Util::printMsgInt("Free SRAM BEFORE NN init => ", Memory::getFreeMemory());
#endif

	// creates the neural network
	bongoNeuralNetwork = new NeuralNetwork(NN_INPUT_SIZE, NN_HIDDEN_LAYERS_SIZES, NN_OUTPUT_SIZE);

#ifdef DEBUG_MEMORY
	Util::printMsgInt("Free SRAM AFTER NN init => ", Memory::getFreeMemory());
#endif

#ifdef SAVE_TRAINING_DATA_SDCARD
	if (!SD.begin(BUILTIN_SDCARD)) {
		Util::printMsg("SD Card failed, or not present");
		return;
	}
	Util::printMsg("card initialized.");
	deleteTrainingDataSDCardFile();
#endif

	lastFFTOutput.resize(NN_INPUT_SIZE);
}

void loop() {

#ifdef TEST_NN
	vector<nn_double> v0;
	vector<nn_double> v1;
	vector<nn_double> v2;
	vector<nn_double> v3;
	vector<nn_double> v4;
	for (size_t i = 0; i < NN_INPUT_SIZE; i++)
	{
		v0.push_back(((rand() % 200) - 100) / 100.0);
		v1.push_back(((rand() % 200) - 100) / 100.0);
		v2.push_back(((rand() % 200) - 100) / 100.0);
		v3.push_back(((rand() % 200) - 100) / 100.0);
		v4.push_back(((rand() % 200) - 100) / 100.0);
	}

	TrainingSet ts0(v0, { 0.0, 0.0 });
	TrainingSet ts1(v1, { 0.0, 1.0 });
	TrainingSet ts2(v2, { 1.0, 1.0 });
	TrainingSet ts3(v3, { });
	TrainingSet ts4(v4, { });

	vector<TrainingSet> tsets{ ts0, ts1, ts2, ts3, ts4 };

	for (size_t i = 0; i < 600; i++)
	{
		bongoNeuralNetwork->feedInputs(tsets[i % 3]);
		bongoNeuralNetwork->propagate();
		bongoNeuralNetwork->feedOutputIdealValues(tsets[i % 3]);
		bongoNeuralNetwork->backpropagate();
	}
	Util::printMsg("\n************************");
	Util::printMsg("*** Training is over ***");
	Util::printMsg("************************\n");

	for (size_t i = 0; i < tsets.size(); i++)
	{
		bongoNeuralNetwork->feedInputs(tsets[i]);
		bongoNeuralNetwork->propagate();
		bongoNeuralNetwork->printOutput();
	}
	while (1);

#else

	while (trainingData.size() < MAX_TRAIN_DATA_SIZE) {
		// adding a training set to the training database when fft is triggered
		updateCurrentBongoRecordingSelection(currentBongoRecording);
		addTrainingSet(currentBongoRecording);
	}

	bongoNeuralNetwork->train(trainingData, TARGET_ERROR, MAX_EPOCHS);


	for (auto& tset : trainingData) {
		bongoNeuralNetwork->feedInputs(tset);
		bongoNeuralNetwork->propagate();
		bongoNeuralNetwork->printOutput();
	}
	while (1) {
		testNeuralNetwork(bongoNeuralNetwork);
		delay(20);
	}

#endif	// TEST_NN
	

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