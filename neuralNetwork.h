//#include "dataEntry.h"
#include <ctime>
#include <fstream>
#include <sstream>
#include "Structure.h"
#define LEARNING_RATE 0.1
#define MOMENTUM 0.9
#define MAX_EPOCHS 1500
#define DESIRED_ACCURACY 100  

using namespace std;

/*******************************************************************************************************************
 *	NEURAL NETWORK CLASS
 *	----------------------------------------------------------------------------------------------------------------
 *	Classic Back-propagation Neural Network ( makes use of gradient descent )
 *	Can toggle between stochastic and batch learning
 *	----------------------------------------------------------------------------------------------------------------
 *******************************************************************************************************************/


//Neural Network Creation
NeuralNetwork::NeuralNetwork()
{
	//create neuron lists
	//--------------------------------------------------------------------------------------------------------

	for (int i = 0; i < nInput; i++)
	{
		inputNeurons[i] = 0;
	}
	//create bias neuron
	inputNeurons[nInput] = -1;

	for (int i = 0; i < nHidden; i++)
	{
		hiddenNeurons[i] = 0;
	}
	//create bias neuron
	hiddenNeurons[nHidden] = -1;


	for (int i = 0; i < nHidden2; i++)
	{
		hiddenNeurons2[i] = 0;
	}
	//create bias neuron
	hiddenNeurons2[nHidden2] = -1;


	for (int i = 0; i < nOutput; i++)
	{
		outputNeurons[i] = 0;
	}


	//create weight lists (include bias neuron weights)
	//--------------------------------------------------------------------------------------------------------
	wInputHidden = new(double*[nInput + 1]);
	for (int i = 0; i <= nInput; i++)
	{
		wInputHidden[i] = new (double[nHidden]);
		for (int j = 0; j < nHidden; j++)
		{
			wInputHidden[i][j] = 0;
		}
	}
	wHiddenHidden2 = new(double*[nHidden + 1]);
	for (int i = 0; i <= nHidden; i++)
	{
		wHiddenHidden2[i] = new (double[nHidden2]);
		for (int j = 0; j < nHidden2; j++)
		{
			wHiddenHidden2[i][j] = 0;
		}
	}
	wHidden2Output = new(double*[nHidden2 + 1]);
	for (int i = 0; i <= nHidden2; i++)
	{
		wHidden2Output[i] = new (double[nOutput]);
		for (int j = 0; j < nOutput; j++)
		{
			wHidden2Output[i][j] = 0;
		}
	}

	//create delta lists
	//--------------------------------------------------------------------------------------------------------

	deltaInputHidden = new(double*[nInput + 1]);
	for (int i = 0; i <= nInput; i++)
	{
		deltaInputHidden[i] = new (double[nHidden]);
		for (int j = 0; j < nHidden; j++) deltaInputHidden[i][j] = 0;
	}

	deltaHiddenHidden2 = new(double*[nHidden + 1]);
	for (int i = 0; i <= nHidden; i++)
	{
		deltaHiddenHidden2[i] = new (double[nHidden2]);
		for (int j = 0; j < nHidden2; j++) deltaHiddenHidden2[i][j] = 0;
	}

	deltaHidden2Output = new(double*[nHidden2 + 1]);
	for (int i = 0; i <= nHidden2; i++)
	{
		deltaHidden2Output[i] = new (double[nOutput]);
		for (int j = 0; j < nOutput; j++) deltaHidden2Output[i][j] = 0;
	}

	//create error gradient storage
	//--------------------------------------------------------------------------------------------------------
	for (int i = 0; i <= nHidden; i++)
	{
		hiddenErrorGradients[i] = 0;
	}

	for (int i = 0; i <= nHidden2; i++)
	{
		hidden2ErrorGradients[i] = 0;
	}

	for (int i = 0; i <= nOutput; i++)
	{
		outputErrorGradients[i] = 0;
	}

	//initialize weights
	//--------------------------------------------------------------------------------------------------------
	initializeWeights();

	//default learning parameters
	//--------------------------------------------------------------------------------------------------------
	learningRate = LEARNING_RATE;
	momentum = MOMENTUM;

	//use stochastic learning by default
	useBatch = false;

	//stop conditions
	//--------------------------------------------------------------------------------------------------------
	maxEpochs = MAX_EPOCHS;
	desiredAccuracy = DESIRED_ACCURACY;

	//NeuralNetwork nn;
	//init random number generator
	//
}

//destructor
NeuralNetwork::~NeuralNetwork()
{
		//delete neurons
		delete[] inputNeurons;
		delete[] hiddenNeurons;
		delete[] hiddenNeurons2;
		delete[] outputNeurons;

		//delete weight storage
		for (int i = 0; i <= nInput; i++) delete[] wInputHidden[i];
		delete[] wInputHidden;

		for (int j = 0; j <= nHidden; j++) delete[] wHiddenHidden2[j];
		delete[] wHiddenHidden2;

		for (int j = 0; j <= nHidden2; j++) delete[] wHidden2Output[j];
		delete[] wHidden2Output;

		//delete delta storage
		for (int i = 0; i <= nInput; i++) delete[] deltaInputHidden[i];
		delete[] deltaInputHidden;

		for (int j = 0; j <= nHidden; j++) delete[] deltaHiddenHidden2[j];
		delete[] deltaHiddenHidden2;

		for (int j = 0; j <= nHidden2; j++) delete[] deltaHidden2Output[j];
		delete[] deltaHidden2Output;

		//delete error gradients
		delete[] hiddenErrorGradients;
		delete[] hidden2ErrorGradients;
		delete[] outputErrorGradients;

		//close log file
		if (logFile.is_open()) logFile.close();
}

void NeuralNetwork::initializeWeights()
{
	srand((unsigned int)time(0));
	//set weights between input and hidden to a random value between -0.5 and 0.5
	//--------------------------------------------------------------------------------------------------------
	for (int i = 0; i <= nInput; i++)
	{
		for (int j = 0; j < nHidden; j++)
		{
			//set weights to random values
			wInputHidden[i][j] = (double)rand() / (RAND_MAX + 1) - 0.5;

			//create blank delta
			deltaInputHidden[i][j] = 0;
		}
	}

	for (int i = 0; i <= nHidden; i++)
	{
		for (int j = 0; j < nHidden2; j++)
		{
			//set weights to random values
			wHiddenHidden2[i][j] = (double)rand() / (RAND_MAX + 1) - 0.5;

			//create blank delta
			deltaHiddenHidden2[i][j] = 0;
		}
	}

	//set weights between input and hidden to a random value between -0.5 and 0.5
	//--------------------------------------------------------------------------------------------------------
	for (int i = 0; i <= nHidden2; i++)
	{
		for (int j = 0; j < nOutput; j++)
		{
			//set weights to random values
			wHidden2Output[i][j] = (double)rand() / (RAND_MAX + 1) - 0.5;

			//create blank delta
			deltaHidden2Output[i][j] = 0;
		}
	}
}


