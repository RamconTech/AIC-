#pragma once

#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <fstream>
#include <sstream>
#define LEARNING_RATE 0.1
#define MOMENTUM 0.9
#define MAX_EPOCHS 1500
#define DESIRED_ACCURACY 90  

class NeuralNetwork
{
public:
	NeuralNetwork();//constructor
	~NeuralNetwork();//destructor
	//void NeuralNetworkStructure();
	void feedForward(double *inputs);
	void backpropagate(double* desiredValues);
	void runTrainingEpoch(vector<dataEntry*> trainingSet, int trainingSetAccuracy);
	void trainNetwork(vector<dataEntry*> trainingSet, vector<dataEntry*> generalizationSet, vector<dataEntry*> validationSet);
	void initializeWeights();
	void enableLogging(const char* filename, int resolution);
	void updateWeights();
	double getSetAccuracy(vector<dataEntry*> set);
	double getSetMSE(vector<dataEntry*> set);
	double getHiddenErrorGradient(int j);
	double getHidden2ErrorGradient(int j);
	double* feedInput(double* inputs);
	void resetWeights(){ initializeWeights(); }
	void setLearningParameters(double lr, double m){ learningRate = lr; momentum = m; }
	void setMaxEpochs(int max){ maxEpochs = max; }
	void setDesiredAccuracy(float d){ desiredAccuracy = d; }
	void useBatchLearning(){ useBatch = true; }
	void useStochasticLearning(){ useBatch = false; }

private:
	//learning parameters
	double learningRate;					// adjusts the step size of the weight update	
	double momentum;						// improves performance of stochastic learning (don't use for batch)

	//number of neurons
	int nInput=16;
	int nHidden=8;
	int nHidden2=8;
	int nOutput=1;

	//change to weights
	double** deltaInputHidden;
	double** deltaHiddenHidden2;
	double** deltaHidden2Output;

	//neurons layers
	double* inputNeurons = new(double[nInput+1]);
	double* hiddenNeurons = new(double[nHidden + 1]);
	double* hiddenNeurons2 = new(double[nHidden2 + 1]);
	double* outputNeurons = new(double[nOutput+1]);

	//weights
	double** wInputHidden;
	double** wHiddenHidden2;
	double** wHidden2Output;

	long epoch=0;
	bool logResults=false;
	fstream logFile;
	int logResolution=10;
	int lastEpochLogged=-10;

	//accuracy stats per epoch
	double trainingSetAccuracy=0;
	double validationSetAccuracy=0;
	double generalizationSetAccuracy=0;
	double trainingSetMSE=0;
	double validationSetMSE=0;
	double generalizationSetMSE=0;

	//epoch counter
	long maxEpochs=1000;

	//accuracy required
	double desiredAccuracy=100;
	
	//error gradients
	double* hiddenErrorGradients= new(double[nHidden + 1]);
	double* hidden2ErrorGradients = new(double[nHidden2 + 1]);
	double* outputErrorGradients = new(double[nOutput + 1]);

	//batch learning flag
	bool useBatch=false;
};

#endif