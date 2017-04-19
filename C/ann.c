/*
name:   ann.c
author: Kevin Klein
date:   14/03/2017
descr.: MLP (Multilayer Perceptron), also referred to as a feedforward artificial
        neural network, using Backpropagation for supervised learning
comp:   gcc -o ann ann.c -lm  
*/

//#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>


// 3 input neurons
#define numInputs 12
// number of neorons in hidden layer
#define numHidden 120
// 4 sets of inputs paired with target outputs
#define numPatterns 4
// XOR problem:
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
void WeightChangesHO();
void WeightChangesIH();
void calcOverallError();
void displayResults();
double getRand();

// variables
int patNum = 0;
double errThisPat = 0.0;
double outPred = 0.0;
double RMSerror = 0.0;

// the outputs of the hidden neurons
double hiddenVal[numHidden];

// the weights
double weightsIH[numInputs][numHidden];
double weightsHO[numHidden];

// the data
int trainInputs[numPatterns][numInputs];
int trainOutput[numPatterns];


//==============================================================
//************** function definitions **************************
//==============================================================

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
   // the output neuron is linear
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
// adjust the weights hidden-output
void WeightChangesHO(void)
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
// adjust the weights input-hidden
void WeightChangesIH(void)
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
// generates a random number
double getRand(void)
{
 return ((double)rand()) / (double)RAND_MAX;
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

//************************************
// read in the data
void initData(void)
{
    printf("initialising data\n");

    // the data here is the XOR data
    // it has been rescaled to the range
    // [-1][1]
    // an extra input valued 1 is also added
    // to act as the bias
    // the output must lay in the range -1 to 1

    trainInputs[0][0]  = 1;
    trainInputs[0][1]  = -1;
    trainInputs[0][2]  = 1;    //bias
    trainOutput[0] = 1;

    trainInputs[1][0]  = -1;
    trainInputs[1][1]  = 1;
    trainInputs[1][2]  = 1;       //bias
    trainOutput[1] = 1;

    trainInputs[2][0]  = 1;
    trainInputs[2][1]  = 1;
    trainInputs[2][2]  = -1;        //bias
    trainOutput[2] = -1;

    trainInputs[3][0]  = -1;
    trainInputs[3][1]  = -1;
    trainInputs[3][2]  = -1;     //bias
    trainOutput[3] = -1;
}

//************************************
// display results
void displayResults(void)
{
  printf("---------------------------------- \n");
  for(int i = 0; i < numPatterns; i++)
  {
     patNum = i;
     //trainNet();
     printf("trainOutput = %d outPred = %f\n", trainOutput[patNum], outPred);
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
         //trainNet();
         RMSerror = RMSerror + (errThisPat * errThisPat);
        }
     RMSerror = RMSerror / numPatterns;
     RMSerror = sqrt(RMSerror);
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

 // train the network
    for(int j = 0; j <= numEpochs; j++)
    {
        for(int i = 0; i < numPatterns; i++)
        {
          // select a pattern at random
          patNum = rand() % numPatterns;

          // calculate the current network output
          // and error for this pattern
          trainNet();

          // change network weights
          WeightChangesHO();
          WeightChangesIH();
        }

        // display the overall network error
        // after each epoch
        calcOverallError();

       printf("errThisPat %f \n", errThisPat);
       printf("epoch = %d RMS Error = %f\n", j, RMSerror);
    }

 // training has finished
 // display the results
 displayResults();

 return 0;
}
