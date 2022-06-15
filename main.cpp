//standard libraries
#include <iostream>
#include <sstream>

//custom includes
#include "dataEntry.h"
#include "dataReader.h"
#include "FeedForward.h"


//use standard namespace
using namespace std;

void NeuralNetwork::runTrainingEpoch(vector<dataEntry*> trainingSet, int trainingSetAccuracy)
{
	NeuralNetwork nn;

	//incorrect patterns
	double incorrectPatterns = 0;
	double mse = 0;

	//for every training pattern
	for (int tp = 0; tp < (int)trainingSet.size(); tp++)
	{
		//feed inputs through network and backpropagate errors
		feedForward(trainingSet[tp]->pattern);
		backpropagate(trainingSet[tp]->target);

		//pattern correct flag
		bool patternCorrect = true;

		//check all outputs from neural network against desired values
		for (int k = 0; k < nn.nOutput; k++)
		{
			//pattern incorrect if desired and output differ
			if (getRoundedOutputValue(nn.outputNeurons[k]) != trainingSet[tp]->target[k]) patternCorrect = false;

			//calculate MSE
			mse += pow((nn.outputNeurons[k] - trainingSet[tp]->target[k]), 2);
		}

		//if pattern is incorrect add to incorrect count
		if (!patternCorrect) incorrectPatterns++;

	}//end for

	//if using batch learning - update the weights
	if (nn.useBatch) updateWeights();

	//update training accuracy and MSE
	trainingSetAccuracy = 100 - (incorrectPatterns / trainingSet.size() * 100);
	nn.trainingSetMSE = mse / (nn.nOutput * trainingSet.size());
}



void NeuralNetwork::enableLogging(const char* filename, int resolution)
{
	//NeuralNetwork nn;
	//create log file 
	if (!logFile.is_open())
	{
		logFile.open(filename, ios::out);

		if (logFile.is_open())
		{
			//write log file header
			logFile << "Epoch,Training Set Accuracy, Generalization Set Accuracy,Training Set MSE, Generalization Set MSE" << endl;

			//enable logging
			logResults = true;

			//resolution setting;
			logResolution = resolution;
			lastEpochLogged = -resolution;
		}
	}
}


void main()
{	
	//create data set reader
	dataReader d;
	NeuralNetwork nn;

	vector<dataEntry*> trainingSet; 
	vector<dataEntry*> generalizationSet; 
	vector<dataEntry*> validationSet;

	//load data file
	/*d.loadDataFile("1.jpg", 16, 1);
	d.loadDataFile("2.jpg", 16, 1);
	d.loadDataFile("3.jpg", 16, 1);
	d.loadDataFile("4.jpg", 16, 1);
	d.loadDataFile("5.jpg", 16, 1);
	d.loadDataFile("6.jpg", 16, 1);
	d.loadDataFile("7.jpg", 16, 1);
	d.loadDataFile("8.jpg", 16, 1);
	d.loadDataFile("9.jpg", 16, 1);*/
	d.loadDataFile("vowel-recognition.csv", 16, 1);
	d.setCreationApproach(STATIC);

	//create neural network
	//nn.neuralNetwork(16, 8, 8, 1);
    nn.enableLogging("trainingResults.csv",1);
	//feedForward(16, 8, 8, 1);
	//runTrainingEpoch();
	//backpropagate(double* desiredValues, 1, 8, 8, 16, 0.1, 0.9);
	//getHiddenErrorGradient(int j, 8);
	//getHidden2ErrorGradient(int j, 1);
	//getSetAccuracy(vector<dataEntry*> set, int nOutput);
	//getSetMSE(vector<dataEntry*> set, int nOutput);
	//feedInput(double* inputs);
    nn.setLearningParameters(0.1, 0.9);
	nn.setDesiredAccuracy(100);
	nn.setMaxEpochs(1000);
	
	dataSet* dset;

	//dataset
	for (int i = 0; i < d.nDataSets(); i++)
	{
		dset = d.getDataSet();
		nn.trainNetwork(trainingSet, generalizationSet, validationSet);
		//nn.trainNetwork(dset,);
	}
	
	cout << "-- END OF PROGRAM --" << endl;
	char c; cin >> c;
}



double* NeuralNetwork::feedInput(double* inputs)
{
	//feed data into the network
	feedForward(inputs);

	//return results
	return outputNeurons;
}
void NeuralNetwork::trainNetwork(vector<dataEntry*> trainingSet, vector<dataEntry*> generalizationSet, vector<dataEntry*> validationSet)
{
		//NeuralNetwork Train
		cout << endl << " Neural Network Training Starting: " << endl
			<< "==========================================================================" << endl
			<< " LR: " << learningRate << ", Momentum: " << momentum << ", Max Epochs: " << maxEpochs << endl
			<< " " << nInput << " Input Neurons, " << nHidden << " Hidden Neurons, " << nHidden2 << " Hidden2 Neurons, " << nOutput << " Output Neurons " << endl
			<< "==========================================================================" << endl << endl;

		//reset epoch and log counters
		epoch = 0;
		lastEpochLogged = -logResolution;

		//train network using training dataset for training and generalization dataset for testing
		//--------------------------------------------------------------------------------------------------------
		while ((trainingSetAccuracy < desiredAccuracy || generalizationSetAccuracy < desiredAccuracy) && epoch < maxEpochs)
		{
			//store previous accuracy
			double previousTAccuracy = trainingSetAccuracy;
			double previousGAccuracy = generalizationSetAccuracy;

			//use training set to train network
			runTrainingEpoch(trainingSet, 100);

			//get generalization set accuracy and MSE
			generalizationSetAccuracy = getSetAccuracy(generalizationSet);
			generalizationSetMSE = getSetMSE(generalizationSet);

			//Log Training results
			if (logResults && logFile.is_open() && (epoch - lastEpochLogged == logResolution))
			{
				logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl;
				lastEpochLogged = epoch;
			}

			//print out change in training /generalization accuracy (only if a change is greater than a percent)
			if (ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy))
			{
				cout << "Epoch :" << epoch;
				cout << " TSet Acc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE;
				cout << " GSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << endl;
			}

			//once training set is complete increment epoch
			epoch++;

		}//end while

		//get validation set accuracy and MSE
		validationSetAccuracy = getSetAccuracy(validationSet);
		validationSetMSE = getSetMSE(validationSet);

		//log end
		logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl << endl;
		logFile << "Training Complete!!! - > Elapsed Epochs: " << epoch << " Validation Set Accuracy: " << validationSetAccuracy << " Validation Set MSE: " << validationSetMSE << endl;

		//out validation accuracy and MSE
		cout << endl << "Training Complete!!! - > Elapsed Epochs: " << epoch << endl;
		cout << " Validation Set Accuracy: " << validationSetAccuracy << endl;
		cout << " Validation Set MSE: " << validationSetMSE << endl << endl;
}