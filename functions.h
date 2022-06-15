#include "Structure.h"
#include "math.h"
#include "dataEntry.h"

inline double activationFunction (double x)
{
		return 1 / (1 + exp(-x));
}

inline double getOutputErrorGradient(double desiredValue, double outputValue)
{
		return outputValue * (1 - outputValue) * (desiredValue - outputValue);
}

int getRoundedOutputValue(double x)
{
        if (x < 0.1) return 0;
		else if (x > 0.9) return 1;
		else return -1;
}

