/*
name:    ann.c
author:  Kevin Klein
date:    14/03/2017
descr.:  MLP (Multilayer Perceptron) using Backpropagation for
		 supervised learning to approximate XOR problem 
compile: gcc -o ann ann.c -lm  
*/

//#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

clock_t start, end;
double cpu_time_used;

// 3 input neurons
#define numInputs 3
// 90 neurons in hidden layer
#define numHidden 90
// 4 sets of inputs paired with target outputs
#define numPatterns 4
// XOR patterns:
// 0,0 -> 0
// 0,1 -> 1
// 1,0 -> 1
// 1,1 -> 0

// number of iterations
const int numEpochs = 5000;

// logistic regression values
const double LR_IH = 0.7;
const double LR_HO = 0.07;


// functions
void initWeights();
void initData();
void trainNet();
void weightChangesHO();
void weightChangesIH();
void calcOverallError();
void displayResults();
double getRand();

// variables
int patNum = 0;
double errThisPat = 0.0;
double outPred = 0.0;
double RMSerror = 0.0;

// ouput value in hidden layer
double hiddenVal[numHidden];

// weights
double weightsIH[numInputs][numHidden];
double weightsHO[numHidden];

// input and output data
int trainInputs[numPatterns][numInputs];
int trainOutput[numPatterns];


//==============================================================
//************** function definitions **************************
//==============================================================

//************************************
// generates a random number
double getRand(void)
{
	return ((double)rand()) / (double)RAND_MAX;
}

//************************************
// read in the data
void initData(void)
{
    printf("initialising data\n");

    // the data here is the XOR input and output data
    // an extra input valued 1 is also added
    // to act as the bias

    // 0 XOR 0 = 0
    trainInputs[0][0] = -1;
    trainInputs[0][1] = -1;
    trainInputs[0][2] = -1; // bias
    trainOutput[0] = -1;

	// 0 XOR 1 = 1
    trainInputs[1][0] = -1;
    trainInputs[1][1] = 1;
    trainInputs[1][2] = 1; // bias
    trainOutput[1] = 1;

	// 1 XOR 0 = 1
    trainInputs[2][0] = 1;
    trainInputs[2][1] = -1;
    trainInputs[2][2] = 1; // bias
    trainOutput[2] = 1;

	// 1 XOR 1 = 0
    trainInputs[3][0] = 1;
    trainInputs[3][1] = 1;
    trainInputs[3][2] = -1; // bias
    trainOutput[3] = -1;
}

//************************************
// set weights to random numbers 
void initWeights(void)
{

 for(int j = 0; j < numHidden; j++)
 {
    weightsHO[j] = (getRand() - 0.5) / 2;

    for(int i = 0; i < numInputs; i++)
    {
     weightsIH[i][j] = (getRand() - 0.5) / 5;
     //printf("Weight = %f\n", weightsIH[i][j]);
    }
  }

}

//***********************************
// calculates the network output
void trainNet(void)
{
    // calculate the outputs of the hidden neurons
    int i = 0;
    for(i = 0; i < numHidden; i++)
    {
    	hiddenVal[i] = 0.0;

        for(int j = 0; j < numInputs; j++)
        {
           // printf("patNum: %i \n", patNum);
           hiddenVal[i] = hiddenVal[i] + (trainInputs[patNum][j] * weightsIH[j][i]);
           // printf("hiddenVal[i]: %f \n", hiddenVal[i]);
        }

        hiddenVal[i] = tanh(hiddenVal[i]);
        // printf("tanh(hiddenVal[i]): %f \n", hiddenVal[i]);
    }

   // calculate the output of the network
   outPred = 0.0;

   for(i = 0; i < numHidden; i++)
   {
    outPred = outPred + hiddenVal[i] * weightsHO[i];
    // printf("outPred %f \n", outPred);
   }
    // calculate the error
    errThisPat = outPred - trainOutput[patNum];
}


//************************************
// adjust the weights input-hidden
void weightChangesIH(void)
{
  for(int i = 0; i < numHidden; i++)
  {
     for(int k = 0; k < numInputs; k++)
     {
        double x = 1 - (hiddenVal[i] * hiddenVal[i]);
        x = x * weightsHO[i] * errThisPat * LR_IH;
        x = x * trainInputs[patNum][k];
        double weightChange = x;
        weightsIH[k][i] = weightsIH[k][i] - weightChange;
     }
  }
}


//************************************
// adjust the weights hidden-output
void weightChangesHO(void)
{
   for(int k = 0; k < numHidden; k++)
   {
    double weightChange = LR_HO * errThisPat * hiddenVal[k];
    weightsHO[k] = weightsHO[k] - weightChange;

    //regularisation on the output weights
    if (weightsHO[k] < -5)
    {
     	weightsHO[k] = -5;
    }
    else if (weightsHO[k] > 5)
    {
     	weightsHO[k] = 5;
    }
   }
 }


//************************************
// calculate the overall error
void calcOverallError(void)
{
    RMSerror = 0.0;
    for(int i = 0; i < numPatterns; i++)
    {
        patNum = i;
        trainNet();
        RMSerror = RMSerror + (errThisPat * errThisPat);
    }

    RMSerror = RMSerror / numPatterns;
    RMSerror = sqrt(RMSerror);
}


//************************************
// display results
void displayResults(void)
{
  printf("---------------------------------- \n");
  for(int i = 0; i < numPatterns; i++)
  {
     patNum = i;
     trainNet();
     printf("target output = %d, actual output = %f\n", trainOutput[patNum], outPred);
  }
}


//==============================================================
//********** THIS IS THE MAIN PROGRAM **************************
//==============================================================

int main(void)
{
 // seed random number function
 srand(time(NULL));

 // initiate the weights
 initWeights();

 // load in the data
 initData();

 // start timer
 start = clock();
 
 // train the network
    for(int j = 0; j <= numEpochs; j++)
    {
        for(int i = 0; i < numPatterns; i++)
        {
          // select one of the input-output patterns for learning
          patNum = rand() % numPatterns;

          // calculate output and error for this pattern
          trainNet();

          // change the weights
          weightChangesHO();
          weightChangesIH();
        }

        // display the overall network error
        calcOverallError();

        // printf("errThisPat %f \n", errThisPat);
        printf("epoch = %d RMS Error = %f\n", j, RMSerror);
    }

    // training has finished
 	end = clock();

 	// display the results
 	displayResults();
 	// time elapsed
 	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
 	printf("----------------------------------\n");
 	printf("cpu time: %f sec \n", cpu_time_used);
 	printf("----------------------------------\n");

 return 0;
}
