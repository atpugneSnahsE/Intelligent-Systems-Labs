# IS-Lab2 – Multi-Layer Perceptron Training

This repository contains MATLAB implementations of a Multi-Layer Perceptron (MLP) trained with backpropagation for function and surface approximation, following the IS-Lab2 laboratory task description.[file:1]

## Overview

The project implements:
- A 1D MLP approximator with one input and one output that learns a nonlinear target function defined using sine terms.
- An extended 2D “surface approximation” MLP with two inputs and one output using the same structural constraints (single hidden layer, nonlinear hidden activation, linear output, backpropagation).[file:1]

## Task Description

The implementation follows the IS-Lab2 assignment on training an MLP approximator.

### Main task

Implement a program that estimates the coefficients (weights and biases) of a multilayer perceptron acting as a function approximator with the following structure:[file:1]

- **Input**: one input variable, using 20 input vectors \(X\) with values in the range \([0,1]\), e.g. `x = 0.1:1/22:1;`.
- **Output**: one output variable, whose desired response is computed from the target function  
  \[
  d(x) = \frac{1 + 0.6 \sin(2 \pi x / 0.7) + 0.3 \sin(2 \pi x)}{2}.
  \]
  The MLP should approximate this function via learned weights, not by hard-coding the analytical formula.[file:1]  
- **Hidden layer**: a single hidden layer with 4–8 neurons, using sigmoidal or hyperbolic tangent activation functions.[file:1]  
- **Output neuron**: linear activation function.[file:1]  
- **Training algorithm**: backpropagation.[file:1]  

### Additional task

Extend the solution to a surface approximation problem with:

- Two input variables.
- One output variable.
- The same general MLP structure (single hidden layer, nonlinear hidden activation, linear output, backpropagation).

![MLP approximation results](path/to/your/image.png)

## MLP Architecture

- **Input layer**  
  - 1 neuron for the main 1D task, or 2 neurons for the surface task.

- **Hidden layer**  
  - 6 neurons with sigmoid or tanh activation functions.

- **Output layer**  
  - 1 neuron with a linear activation function.

- **Training**  
  - Backpropagation with gradient-based weight and bias updates on the approximation error.


