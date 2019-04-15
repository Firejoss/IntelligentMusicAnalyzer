#pragma once

//#define TEST_NN

//#define FFT1024		// if commented out, FFT 256 resolution will be set by default
#ifdef FFT1024
#undef FFT256
#else
#define FFT256 true
#endif

/* ----- Neural network defines & variables ----- */
// imposed number of NN inputs given the data vector provided by the fft
#ifdef FFT256
#define NN_INPUT_SIZE					128
#elif FFT1024
#define NN_INPUT_SIZE					512
#endif

#define NN_HIDDEN_LAYERS_SIZES			{ 64, 128, 64, 64, 16 }	
#define NN_OUTPUT_SIZE					2

//#define SAVE_TRAINING_DATA_SDCARD

#define FILENAME_TRAIN_DATA             "nn_train1.dat"
#define MAX_TRAIN_DATA_SIZE				60		// *** TRAINING SIZE ***
#define FFT_SUM_ADD_TRAININGSET_THRES	0.15

#define MAX_EPOCHS						120
#define TARGET_ERROR					0.1	// *** TARGET ERROR ***


// defines which bongo is to be played 
// for NN training and expected output vector
#define BONGO_NOISE_RECORDING			{0, 0}
#define BONGO_1_RECORDING				{0, 1}
#define BONGO_2_RECORDING				{1, 0}

// ----- GENERAL DEFINES -----
#define ON								1
#define OFF								0
#define G_SWITCH						ON
#define R_PIN							6
#define G_PIN							7
#define B_PIN							8 
#define MIC_PIN							PIN_A3
#define BONGO_SELECT_BTN_PIN_1			15
#define BONGO_SELECT_BTN_PIN_2			20
#define DEBUG_SERIAL
#define DEBUG_SDCARD
//#define DEBUG_BONGO_SELECTION

// prevent compile time error
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