#include "functions.h"

void NeuralNetwork::feedForward(double *inputs)
{
	cout << "good";
	//set input neurons to input values
	for (int i = 0; i < nInput; i++) inputNeurons[i] = inputs[i];

	//Calculate Hidden Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j < nHidden; j++)
	{
		//clear value
		hiddenNeurons[j] = 0;

		//get weighted sum of inputs and bias neuron
		for (int i = 0; i <= nInput; i++)
		{
			hiddenNeurons[j] += inputNeurons[i] * wInputHidden[i][j];
		}

		//set to result of sigmoid
		hiddenNeurons[j] = activationFunction(hiddenNeurons[j]);
	}

	for (int j = 0; j < nHidden2; j++)
	{
		//clear value
		hiddenNeurons2[j] = 0;

		//get weighted sum of inputs and bias neuron
		for (int i = 0; i <= nHidden; i++)
		{
			hiddenNeurons2[j] += hiddenNeurons[i] * wHiddenHidden2[i][j];
		}

		//set to result of sigmoid
		hiddenNeurons2[j] = activationFunction(hiddenNeurons2[j]);
	}

	//Calculating Output Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------
	for (int k = 0; k < nOutput; k++)
	{
		//clear value
		outputNeurons[k] = 0;

		//get weighted sum of inputs and bias neuron
		for (int j = 0; j <= nHidden2; j++)
		{
			outputNeurons[k] += hiddenNeurons2[j] * wHidden2Output[j][k];
		}

		//set to result of sigmoid
		outputNeurons[k] = activationFunction(outputNeurons[k]);
	}
}

double NeuralNetwork::getSetAccuracy(vector<dataEntry*> set)
{
	double incorrectResults = 0;

	//for every training input array
	for (int tp = 0; tp < (int)set.size(); tp++)
	{
		//feed inputs through network and backpropagate errors
		feedForward(set[tp]->pattern);

		//correct pattern flag
		bool correctResult = true;

		//check all outputs against desired output values
		for (int k = 0; k < nOutput; k++)
		{
			//set flag to false if desired and output differ
			if (getRoundedOutputValue(outputNeurons[k]) != set[tp]->target[k]) correctResult = false;
		}

		//inc training error for a incorrect result
		if (!correctResult) incorrectResults++;

	}//end for

	//calculate error and return as percentage
	return 100 - (incorrectResults / set.size() * 100);
}
//feed forward set of patterns and return MSE

double NeuralNetwork::getSetMSE(vector<dataEntry*> set)
{
	double mse = 0;

	//for every training input array
	for (int tp = 0; tp < (int)set.size(); tp++)
	{
		//feed inputs through network and backpropagate errors
		feedForward(set[tp]->pattern);

		//check all outputs against desired output values
		for (int k = 0; k < nOutput; k++)
		{
			//sum all the MSEs together
			mse += pow((outputNeurons[k] - set[tp]->target[k]), 2);
		}

	}//end for

	//calculate error and return as percentage
	return mse / (nOutput * set.size());
}

void NeuralNetwork::backpropagate(double* desiredValues)
{
	cout << "good";
	//modify deltas between hidden and output layers
	//--------------------------------------------------------------------------------------------------------
	for (int k = 0; k < nOutput; k++)
	{
		//get error gradient for every output node
		outputErrorGradients[k] = getOutputErrorGradient(desiredValues[k], outputNeurons[k]);

		//for all nodes in hidden layer and bias neuron
		for (int j = 0; j <= nHidden2; j++)
		{
			//calculate change in weight
			if (!useBatch) deltaHidden2Output[j][k] = learningRate * hiddenNeurons[j] * outputErrorGradients[k] + momentum * deltaHidden2Output[j][k];
			else deltaHidden2Output[j][k] += learningRate * hiddenNeurons[j] * outputErrorGradients[k];
		}
	}

	//modify deltas between input and hidden layers
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j < nHidden2; j++)
	{
		//get error gradient for every hidden node
		hidden2ErrorGradients[j] = getHidden2ErrorGradient(j);

		//for all nodes in input layer and bias neuron
		for (int i = 0; i <= nHidden; i++)
		{
			//calculate change in weight 
			if (!useBatch) deltaHiddenHidden2[i][j] = learningRate * hiddenNeurons[i] * hiddenErrorGradients[j] + momentum * deltaHiddenHidden2[i][j];
			else deltaHiddenHidden2[i][j] += learningRate * hiddenNeurons[i] * hidden2ErrorGradients[j];
		}
	}
	for (int j = 0; j < nHidden; j++)
	{
		//get error gradient for every hidden node
		hiddenErrorGradients[j] = getHiddenErrorGradient(j);

		//for all nodes in input layer and bias neuron
		for (int i = 0; i <= nInput; i++)
		{
			//calculate change in weight 
			if (!useBatch) deltaInputHidden[i][j] = learningRate * inputNeurons[i] * hiddenErrorGradients[j] + momentum * deltaInputHidden[i][j];
			else deltaInputHidden[i][j] += learningRate * inputNeurons[i] * hiddenErrorGradients[j];
		}
	}
	//if using stochastic learning update the weights immediately
	if (!useBatch) updateWeights();
}

void NeuralNetwork::updateWeights()
{
	//input -> hidden weights
	//--------------------------------------------------------------------------------------------------------
	for (int i = 0; i <= nInput; i++)
	{
		for (int j = 0; j < nHidden; j++)
		{
			//update weight
			wInputHidden[i][j] += deltaInputHidden[i][j];

			//clear delta only if using batch (previous delta is needed for momentum
			if (useBatch) deltaInputHidden[i][j] = 0;
		}
	}
	for (int j = 0; j <= nHidden; j++)
	{
		for (int k = 0; k < nHidden2; k++)
		{
			//update weight
			wHiddenHidden2[j][k] += deltaHiddenHidden2[j][k];

			//clear delta only if using batch (previous delta is needed for momentum)
			if (useBatch)deltaHiddenHidden2[j][k] = 0;
		}
	}
	//hidden -> output weights
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j <= nHidden2; j++)
	{
		for (int k = 0; k < nOutput; k++)
		{
			//update weight
			wHidden2Output[j][k] += deltaHidden2Output[j][k];

			//clear delta only if using batch (previous delta is needed for momentum)
			if (useBatch)deltaHidden2Output[j][k] = 0;
		}
	}
}

double NeuralNetwork::getHiddenErrorGradient(int j)
{
	//get sum of hidden->output weights * output error gradients
	double weightedSum = 0;
	for (int k = 0; k < nHidden2; k++)
	{
		weightedSum += wHiddenHidden2[j][k] * hiddenErrorGradients[k];
	}

	//return error gradient
	return hiddenNeurons[j] * (1 - hiddenNeurons[j]) * weightedSum;
}

double NeuralNetwork::getHidden2ErrorGradient(int j)
{
	//get sum of hidden->output weights * output error gradients
	double weightedSum = 0;
	for (int k = 0; k < nOutput; k++) weightedSum += wHidden2Output[j][k] * outputErrorGradients[k];

	//return error gradient
	return hiddenNeurons2[j] * (1 - hiddenNeurons2[j]) * weightedSum;
}
