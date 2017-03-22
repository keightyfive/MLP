/*
name:	ann.c
author: Kevin Klein
date:	14/03/2017
descr.:	MLP (Multilayer Perceptron), also referred to as a feedforward artificial
 		neural network, using Backpropagation for supervised learning
*/

//-------------------------------------------------------------------------
// preprocessor directives
//-------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//-------------------------------------------------------------------------
// preprocessor macros
//-------------------------------------------------------------------------
// note that neurons are processing elements, that's why the nodes in the 
// input layer are not really considered as neurons... 
#define MAX_NO_OF_LAYERS 3 // no of layers, excluding input layer
#define MAX_NO_OF_INPUTS 2 // no of input nodes in input layer
#define MAX_NO_OF_NEURONS 10
#define MAX_NO_OF_WEIGHTS 31
#define MAX_NO_OF_OUTPUTS 2 // no of neurons in output layer

//-------------------------------------------------------------------------
// structure for the neural network itself, its layers and its neurons...
//-------------------------------------------------------------------------
static struct neural_net
{
	int no_of_inputs;
	int no_of_layers;
	int no_of_batch_changes;
	double *inputs;
	double *outputs;
	struct layer *layers;
} ann;

struct layer
{
	int no_of_neurons;
	struct neuron *neurons;
};

struct neuron
{
	int no_of_inputs;
	double *inputs;
	double *output;
	double *weights;
	double *old_weights;
	double *batch_weight_changes;
	double act_funct_flatness;
	double threshold;
	double old_threshold;
	double batch_threshold_change;
	double *error;
	char axon_family;
};

//-------------------------------------------------------------------------
// init static 1-dimensional arrays with preprocessor macros values
//-------------------------------------------------------------------------
static double net_inputs[MAX_NO_OF_INPUTS];
static struct layer net_layers[MAX_NO_OF_LAYERS];
static struct neuron net_neurons[MAX_NO_OF_NEURONS];
static double neuron_outputs[MAX_NO_OF_NEURONS];
static double net_errors[MAX_NO_OF_NEURONS];
static double net_weights[MAX_NO_OF_WEIGHTS];
static double net_old_weights[MAX_NO_OF_WEIGHTS];
static double batch_weight_changes[MAX_NO_OF_WEIGHTS];

//-------------------------------------------------------------------------
// functions declarations
//-------------------------------------------------------------------------
double get_rand();
void create_ann(int, int *, int *, char *, double *, int);
// TODO:
void feed_inputs(double *);
void update_output(void);
double *get_outputs();
void train(double, double, int, double *);
void apply_batch_cumulations(double, double);
int load_net(char *);
int save_net(char *);

//-------------------------------------------------------------------------
// functions definitions
//-------------------------------------------------------------------------
// get random values for init
double get_rand()
{
	return(((double)rand() * 2) / ((double)RAND_MAX + 1)) - 1;
}

// create the artificial neural network
void create_ann(int no_of_layers, int *no_of_neurons, int *no_of_inputs,
				char *axon_families, double *act_funct_flatnesses, int init_weights)
{
	// loop variables
	int i, j, counter_1, counter_2, counter_3, counter_4 = 0;

	// total number of neurons and weights
	int total_no_of_neurons = 0;
	int total_no_of_weights = 0;

	// assign static array values to neural_net structure members
	// inputs
	ann.inputs = net_inputs;
	ann.no_of_inputs = no_of_neurons[0];
	// layers
	ann.layers = net_layers;
	ann.no_of_layers = no_of_layers;

	//printf("ann.inputs = %p \n", &ann.inputs); // address
	//printf("ann.inputs = %f \n", *ann.inputs); // value

	// get total number of neurons
	for(i = 0; i < ann.no_of_layers; i++)
	{
		total_no_of_neurons += no_of_neurons[i]; 
	}

	// test print
	printf("total_no_of_neurons = %d \n", total_no_of_neurons);

	// init total number of neurons with 0, excluding input layer
	for(i = 0; i < total_no_of_neurons; i++)
	{
		neuron_outputs[i] = 0;
	}

	// get total number of weights by multiplying no of inputs nodes of one layer
	// with no of neurons in the next layer: (2*5) + (5*3) + (3*2) = 31
	for(i = 0; i < ann.no_of_layers; i++)
	{
		total_no_of_weights += no_of_inputs[i] * no_of_neurons[i];		
	}

	// test print
	printf("total_no_of_weights = %d \n", total_no_of_weights);

	// loop through all neurons in all layers
	for(i = 0; i < ann.no_of_layers; i++)
	{
		for(j = 0; j < no_of_neurons[i]; j++)
		{
			// beginning of output layer
			if(i == ann.no_of_layers - 1 && j == 0)
			{
				ann.outputs = &neuron_outputs[counter_1];
			}

			// init no of inputs, weights, activation functions etc...
			net_neurons[counter_1].output = &neuron_outputs[counter_1];
			net_neurons[counter_1].no_of_inputs = no_of_inputs[i];
			net_neurons[counter_1].weights = &net_weights[counter_2];
			net_neurons[counter_1].batch_weight_changes = &batch_weight_changes[counter_2];
			net_neurons[counter_1].old_weights = &net_old_weights[counter_2];
			net_neurons[counter_1].axon_family = axon_families[i];
			net_neurons[counter_1].act_funct_flatness = act_funct_flatnesses[i];

			// test print
			//printf("net_neurons[counter_1].weights = %f \n", *net_neurons[counter_1].weights);
			
			if(i == 0)
			{
				net_neurons[counter_1].inputs = net_inputs;
			}
			else
			{
				net_neurons[counter_1].inputs = &neuron_outputs[counter_3];
			}

			// init error value
			net_neurons[counter_1].error = &net_errors[counter_1];
			counter_2 += no_of_neurons[i];
			counter_1++;
		}

		net_layers[i].no_of_neurons = no_of_neurons[i];
		net_layers[i].neurons = &net_neurons[counter_4];

		if(i > 0)
		{
			counter_3 += no_of_neurons[i-1];
		}
		counter_4 += no_of_neurons[i];
	}

	// init weights and thresholds with random values
	if(init_weights == 1)
	{
		for(i = 0; i < total_no_of_neurons; i++)
		{
			net_neurons[i].threshold = get_rand();
		}
		
		for(i = 0; i < total_no_of_weights; i++)
		{
			net_weights[i] = get_rand();
		}

		// update weights and threshold
		for(i = 0; i < total_no_of_weights; i++)
		{
			net_old_weights[i] = net_weights[i];
		}

		for(i = 0; i < total_no_of_neurons; i++)
		{
			net_neurons[i].old_threshold = net_neurons[i].threshold;
		}
	}
		// init batch values
		for(i = 0; i < total_no_of_neurons; i++)
		{
			net_neurons[i].batch_threshold_change = 0;
		}

		for(i = 0; i < total_no_of_weights; i++)
		{
			net_neurons[i].batch_weight_changes = 0;
		}
	
		ann.no_of_batch_changes = 0;
} // end void create_ann()

//-------------------------------------------------------------------------
// MAIN FUNCTION
//-------------------------------------------------------------------------
int main()
{
	// input and output nodes
	double inputs[MAX_NO_OF_INPUTS];
	double output_targets[MAX_NO_OF_OUTPUTS];

	// determine create_ann() paramaters
	// no of layers, excluding input layer
	int no_of_layers = 3;
	// no of neurons in each layer, excluding input layer
	int no_of_neurons[] = {5,3,2};
	// no of neurons in each layer, excluding output layer
	int no_of_inputs[] = {2,5,3};
	// activation functions for each layer, excluding input layer
	char axon_families[] = {'g','g','t'};
	// whatever the hell this is...
	double act_funct_flatnesses[] = {1,1,1};

	// call create_ann() function to create artificial neural network 
	create_ann(no_of_layers, no_of_neurons, no_of_inputs, axon_families, act_funct_flatnesses, 1);

	return 0;
}
