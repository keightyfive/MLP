/*
name:    ann.c
author:  Kevin Klein
date:    14/03/2017
descr.:  ANN (Artificial Neural Network) using Backpropagation and
		 supervised learning, learns XOR problem 
compile: gcc -o ann ann.c -lm
*/

//#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// to measure runtime
clock_t start, end;
double cpu_time_used;

// number of input neurons
#define NO_INPUT_NEURONS 3
// number of neurons in hidden layer
#define NO_HIDDEN_NEURONS 8
// 4 sets of inputs paired with their target outputs in XOR problem
#define NO_PATTERNS 4

// number of training epochs (iterations)
const int no_epochs = 200;

// logistic regression values for backpropagation
const double LR_IH = 0.7;
const double LR_HO = 0.07;

// other variables
int pattern_no = 0;
double pattern_error = 0.0;
double actual_output = 0.0;
double rms_error = 0.0;

// vector to store results in hidden layer
double values_hidden_layer[NO_HIDDEN_NEURONS];

// weight matrices
double weights_IH[NO_INPUT_NEURONS][NO_HIDDEN_NEURONS];
double weights_HO[NO_HIDDEN_NEURONS];

// arrays to store training data
int input_data[NO_PATTERNS][NO_INPUT_NEURONS];
int output_data[NO_PATTERNS];

// ann functions
double get_rand();
void init_data();
void init_weights();
void feed_forward();
void backprop();
void calc_error();

// random number for first init of matrices
double get_rand(void)
{
	return ((double)rand()) / (double)RAND_MAX;
}

// init input and output data to learn XOR problem
void init_data(void)
{
    printf("initialising data\n");

    // 0 XOR 0 = 0
    input_data[0][0] = -1;	// input x
    input_data[0][1] = -1; 	// input y
    input_data[0][2] = 1;  	// additional bias neuron
    output_data[0] = -1;	// target output

	// 0 XOR 1 = 1
    input_data[1][0] = -1;
    input_data[1][1] = 1;
    input_data[1][2] = 1;
    output_data[1] = 1;

	// 1 XOR 0 = 1
    input_data[2][0] = 1;
    input_data[2][1] = -1;
    input_data[2][2] = 1;
    output_data[2] = 1;

	// 1 XOR 1 = 0
    input_data[3][0] = 1;
    input_data[3][1] = 1;
    input_data[3][2] = 1;
    output_data[3] = -1;
}

// init weights with random values
void init_weights(void)
{
	for(int j = 0; j < NO_HIDDEN_NEURONS; j++)
 	{
    	weights_HO[j] = (get_rand() - 0.5) / 2;

    	for(int i = 0; i < NO_INPUT_NEURONS; i++)
    	{
     		weights_IH[i][j] = (get_rand() - 0.5) / 5;
    	}
  	}
}

// feed the data forward through the neural network
void feed_forward(void)
{
    int i = 0;
    for(i = 0; i < NO_HIDDEN_NEURONS; i++)
    {
    	values_hidden_layer[i] = 0.0;

    	// matrix multiplication of inputs and weights
        for(int j = 0; j < NO_INPUT_NEURONS; j++)
        {
           values_hidden_layer[i] += input_data[pattern_no][j] * weights_IH[j][i];
        }

        // activation function
        values_hidden_layer[i] = tanh(values_hidden_layer[i]);
    }

   	// value of output neuron
   	actual_output = 0.0;

   	for(i = 0; i < NO_HIDDEN_NEURONS; i++)
   	{
   		actual_output += values_hidden_layer[i] * weights_HO[i];
   	}
    // calculate error for this pattern
    pattern_error = actual_output - output_data[pattern_no];    	
}

// backpropagation algorithm to update the weights
void backprop(void)
{
	// update weights between hidden and output layer
	for(int m = 0; m < NO_HIDDEN_NEURONS; m++)
   	{
   		// update weight matrix
   		double weightChange = LR_HO * pattern_error * values_hidden_layer[m];
    	weights_HO[m] -= weightChange;

    	// regularisation on the output weights
    	if(weights_HO[m] < -5)
    	{
     		weights_HO[m] = -5;
    	}
    	else if(weights_HO[m] > 5)
    	{
     		weights_HO[m] = 5;
    	}
   	}

	// update weights between input and hidden layer
  	for(int i = 0; i < NO_HIDDEN_NEURONS; i++)
  	{
    	for(int k = 0; k < NO_INPUT_NEURONS; k++)
     	{
     		// update weight matrix
        	double weightChange = ((1 - (values_hidden_layer[i] * values_hidden_layer[i])) * weights_HO[i] * pattern_error * LR_IH) * input_data[pattern_no][k];
        	weights_IH[k][i] -= weightChange;
     	}
  	}
}

// calculate the overall error
void calc_error(void)
{
    rms_error = 0.0;

    // calculate error for each pattern
    for(int i = 0; i < NO_PATTERNS; i++)
    {
        pattern_no = i;
        feed_forward();
        rms_error += pattern_error * pattern_error;
    }

    // square root error
    rms_error = sqrt(rms_error / NO_PATTERNS);
}


int main(void)
{
	// seed random number function
 	srand(time(NULL));

 	// initialise inputs and target ouputs
 	init_data();

 	// initiate the weights with random values
 	init_weights();

 	// start timer
 	start = clock();
 
 	// train the neural network
    for(int j = 0; j <= no_epochs; j++)
    {
        for(int i = 0; i < NO_PATTERNS; i++)
        {
        	// select one of the patterns as input and target output
          	pattern_no = rand() % NO_PATTERNS;

          	// feed the data forward
          	feed_forward();

          	// update the weights
          	backprop();
        }

        // calculate the overall network error
        calc_error();

        // display no of epoch and overall error
        printf("epoch = %d RMS Error = %f\n", j, rms_error);
    }

    // training has finished, stop timer
 	end = clock();

 	// display the results
 	printf("---------------------------------- \n");
  	for(int i = 0; i < NO_PATTERNS; i++)
  	{
    	pattern_no = i;
     	feed_forward();
     	printf("target output = %d, actual output = %f\n", output_data[pattern_no], actual_output);
  	}

 	// display time elapsed for trainging
 	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
 	printf("----------------------------------\n");
 	printf("time: %f sec \n", cpu_time_used);
 	printf("----------------------------------\n");

 return 0;
}
