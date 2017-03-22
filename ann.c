/*
name:	ann.c
author: Kevin Klein
date:	14/03/2017
descr.:	MLP (Multilayer Perceptron), also referred to as a feedforward artificial
 		neural network, using Backpropagation for supervised learning
comp: 	gcc -o ann ann.c -lm	
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
#define MAX_NO_OF_NEURONS 10 // 82
#define MAX_NO_OF_WEIGHTS 31 // 1660
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
void feed_inputs(double *);
void update_net_output(void);
// static void update_neuron_output(struct neuron *);
// static double derivative(struct neuron *);
void train_net(double, double, int, double *);
double *get_outputs();
void apply_batch_cumulations(double, double);
// TODO:
int load_net(char *);

//int save_net(char *);

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

// feed inputs to nodes in input layer
void feed_inputs(double *inputs)
{
	int i;

	for(i = 0; i < ann.no_of_inputs; i++)
	{
		net_inputs[i] = inputs[i];
	}
}

static void update_neuron_output(struct neuron * my_neuron)
{
	double activation = 0;
	int i;

	// multiply input with weight for each neuron
	for(i = 0; i < my_neuron->no_of_inputs; i++)
	{
		activation += my_neuron->inputs[i] * my_neuron->weights[i];
	}

	// activate according to threshold
	activation += -1 * my_neuron->threshold;

	double temp;

	// cases for different activation functions logistic, tanh, linear
	switch(my_neuron->axon_family)
	{
		// logistic
		case 'g':
		temp = -activation / my_neuron->act_funct_flatness;

		if(temp > 45)
		{
			*(my_neuron->output) = 0;
		}
		else if(temp < -45)
		{
			*(my_neuron->output) = 1;
		}
		else
		{
			*(my_neuron->output) = 1.0 / (1 + exp(temp));	
		}
		break;

		// tanh
		case 't':
		temp = -activation / my_neuron->act_funct_flatness;

		if(temp > 45)
		{
			*(my_neuron->output) = -1;
		}
		else if(temp < -45)
		{
			*(my_neuron->output) = 1;
		}
		else
		{
			*(my_neuron->output) = (2.0 / (1 + exp(temp))) - 1;
		}
		break;

		// linear
		case 'l':
		*(my_neuron->output) = activation;
		break;

		default:
		break;
	}//end switch
}//end static void update_neuron_output

// update the output for each iteration
void update_net_output()
{
	int i, j;

	// loop through layers and layers
	for(i = 0; i < ann.no_of_layers; i++)
	{
		for(j = 0; j < ann.layers[i].no_of_neurons; j++)
		{
			update_neuron_output(&(ann.layers[i].neurons[j]));
		}		
	}
}

static double derivative(struct neuron * my_neuron)
{
	double temp;
	switch(my_neuron->axon_family)
	{
		// logistic
		case 'g':
			temp = ( *(my_neuron->output) * (1.0 - *(my_neuron->output))) / my_neuron->act_funct_flatness;
		break;
		// tanh
		case 't':
			temp = (1 - pow( *(my_neuron->output), 2)) / (2.0 * my_neuron->act_funct_flatness);
		break;
		// linear
		case 'l':
			temp = 1;
		break;
		default:
			temp = 0;
		break;
	}
	return temp;
}

// function to train ann using batch method  
void train_net(double learning_rate, double momentum_rate, int batch, double *output_targets)
{
	int i, j, k;
	double temp;
	struct layer *curr_layer;
	struct layer *next_layer;

	// calcuate errors
	for(i = ann.no_of_layers - 1; i >= 0; i--)
	{
		curr_layer = &ann.layers[i];

		// output layer
		if(i == ann.no_of_layers -1)
		{
			for(j = 0; j < curr_layer->no_of_neurons; j++)
			{
				*(curr_layer->neurons[j].error) = derivative(&curr_layer->neurons[j]) * (output_targets[j] - *(curr_layer->neurons[j].output));
			}
		}
		// other layers
		else
		{
			next_layer = &ann.layers[i+1];
			for(j = 0; j < curr_layer->no_of_neurons; j++)
			{
				temp = 0;
				for(k = 0; k < next_layer->no_of_neurons; k++)
				{
					temp += *(next_layer->neurons[k].error) * next_layer->neurons[k].weights[j];
				}

			*(curr_layer->neurons[j].error) = derivative(&curr_layer->neurons[j]) * temp;	
			}
		}
	}

	// update weights and thresholds
	double temp_weight;
	for(i = ann.no_of_layers -1; i >= 0; i--)
	{
		curr_layer = &ann.layers[i];
		for(j = 0; curr_layer->no_of_neurons; j++)
		{
			// thresholds
			if(batch == 1)
			{
				curr_layer->neurons[j].batch_threshold_change += *(curr_layer->neurons[j].error) * -1;
			}
			else
			{
				temp_weight = curr_layer->neurons[j].threshold;
				curr_layer->neurons[j].threshold += (learning_rate * *(curr_layer->neurons[j].error) * -1) + (momentum_rate * (curr_layer->neurons[j].threshold - curr_layer->neurons[j].old_threshold));
				curr_layer->neurons[j].old_threshold = temp_weight;
			}

			// weights
			if(batch == 1)
			{
				for(k = 0; k < curr_layer->neurons[j].no_of_inputs; k++)
				{
					curr_layer->neurons[j].batch_weight_changes[k] += *(curr_layer->neurons[j].error) * curr_layer->neurons[j].inputs[k];
				}
			}
			else
			{
				for(k = 0; k < curr_layer->neurons[j].no_of_inputs; k++)
				{
					temp_weight = curr_layer->neurons[j].weights[k];

					curr_layer->neurons[j].weights[k] += (learning_rate * *(curr_layer->neurons[j].error) * curr_layer->neurons[j].inputs[k]) + (momentum_rate * (curr_layer->neurons[j].weights[k] - curr_layer->neurons[j].old_weights[k]));
					curr_layer->neurons[j].old_weights[k] = temp_weight;
				}
			}
		}
	}

	if(batch == 1)
	{
		ann.no_of_batch_changes++;
	}
}//end void train_net()

void apply_batch_cumulations(double learning_rate, double momentum_rate)
{
	int i, j, k;
	struct layer *curr_layer;
	double temp_weight;

	for(i = ann.no_of_layers - 1; i >= 0; i--)
	{
		curr_layer = &ann.layers[i];
		for(j = 0; j < curr_layer->no_of_neurons; j++)
		{
			// thresholds
			temp_weight = curr_layer->neurons[j].threshold;
			curr_layer->neurons[j].threshold += (learning_rate * (curr_layer->neurons[j].batch_threshold_change / ann.no_of_batch_changes)) + (momentum_rate * (curr_layer->neurons[j].threshold - curr_layer->neurons[j].old_threshold));
		
			curr_layer->neurons[j].old_threshold = temp_weight;
			curr_layer->neurons[j].batch_threshold_change = 0;

			// weights
			for(k = 0; k < curr_layer->neurons[j].no_of_inputs; k++)
			{
				temp_weight = curr_layer->neurons[j].weights[k];
				curr_layer->neurons[j].weights[k] += (learning_rate * (curr_layer->neurons[j].batch_weight_changes[k] / ann.no_of_batch_changes)) + (momentum_rate * (curr_layer->neurons[j].weights[k] - curr_layer->neurons[j].old_weights[k]));
				curr_layer->neurons[j].old_weights[k] = temp_weight;
				curr_layer->neurons[j].batch_weight_changes[k] = 0;
			}
		}
	}
}//end void apply_batch_cumulations()

// function to get_output
double *get_outputs()
{
	return ann.outputs;
}

int load_net(char *path)
{
	int i,j,k;
	int temp_int;
	double temp_double;
	char temp_char;

	int no_of_layers;
	int no_of_neurons[MAX_NO_OF_LAYERS];
	int no_of_inputs[MAX_NO_OF_LAYERS];
	char axon_families[MAX_NO_OF_LAYERS];
	double act_funct_flatnesses[MAX_NO_OF_LAYERS];

	FILE *in_file;

	if(!(in_file = fopen(path, "rb")))
	return 1;

	// read from file	
	fread(&temp_int, sizeof(int), 1, in_file);
	no_of_layers = temp_int;

	// read from file
	for(i = 0; i < no_of_layers; i++)
	{	
		fread(&temp_int, sizeof(int), 1, in_file);
		no_of_neurons[i] = temp_int;

		fread(&temp_int, sizeof(int), 1, in_file);
		no_of_inputs[i] = temp_int;

		fread(&temp_int, sizeof(char), 1, in_file);
		axon_families[i] = temp_char;

		fread(&temp_int, sizeof(double), 1, in_file);
		act_funct_flatnesses[i] = temp_double;
	}

	create_ann(no_of_layers, no_of_neurons, no_of_inputs, axon_families, act_funct_flatnesses, 0);

	// weights
	for(i = 0; i < no_of_layers; i++)
	{
		for(j = 0; j < no_of_neurons[i]; j++)
		{
			fread(&temp_double, sizeof(double), 1, in_file);
			ann.layers[i].neurons[j].threshold = temp_double;
			
			for(k = 0; k < no_of_inputs[i]; k++)
			{
				fread(&temp_double, sizeof(double), 1, in_file);
				ann.layers[i].neurons[j].weights[k] = temp_double;
			}
		}
	}

	// close file
	fclose(in_file);

	return 0;
}


//-------------------------------------------------------------------------
// MAIN function
//-------------------------------------------------------------------------
int main(int argc, char *argv[])
{
	// input and output nodes
	double inputs[MAX_NO_OF_INPUTS];
	double output_targets[MAX_NO_OF_OUTPUTS];

	// determine create_ann() paramaters
	// no of layers, excluding input layer
	int no_of_layers = 3;
	// no of neurons in each layer, excluding input layer
	int no_of_neurons[] = {5,3,2}; // {50,30,2}
	// no of neurons in each layer, excluding output layer
	int no_of_inputs[] = {2,5,3}; // {2,50,30}
	// activation functions for each layer, excluding input layer
	char axon_families[] = {'g','g','t'};
	// whatever the hell this is...
	double act_funct_flatnesses[] = {1,1,1};

	// call create_ann() function to create artificial neural network 
	create_ann(no_of_layers, no_of_neurons, no_of_inputs, axon_families, act_funct_flatnesses, 1);

	// train ann using batch method
	int i;
	double temp_total;
	int counter = 0;

	for(i = 0; i < 10000; i++)
	{
		// init input nodes with random values
		inputs[0] = get_rand();
		inputs[1] = get_rand();
		temp_total = inputs[0] + inputs[1];

		// call feed_inputs() function 
		feed_inputs(inputs);

		// call update_output() function
		update_net_output();

		// function we want to approximate
		output_targets[0] = (double)sin(temp_total);
		output_targets[1] = (double)cos(temp_total);

		// train using batch training method
		train_net(0, 0, 1, output_targets);
		counter++;

		if(counter == 100)
		{
		// train using batch training
		// don't update weights, just cumulate them
			apply_batch_cumulations(.8,.8);
			counter = 0;
		}
	}

	// test it
	double *outputs;
	printf("Sin Target \t Output \t Cos Target \t Output\n");
	printf("---------- \t -------- \t ---------- \t --------\n");
	for(i = 0; i > 50; i++)
	{
		inputs[0] = get_rand();
		inputs[1] = get_rand();

		temp_total= inputs[0] + inputs[1];
		feed_inputs(inputs);
		update_net_output();
		outputs = get_outputs();

		printf("%f \t %f \t %f \t %f \n", sin(temp_total), outputs[0], cos(temp_total), outputs[1]);
	}
	getchar();

	return 0;
}
