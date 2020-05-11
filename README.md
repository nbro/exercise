# README

## Problem

Implement a genetic algorithm to maximize the numeric function 

    (2*x*z * exp(-x) - 2*y^3 + y^2 - 3*z^3)
    2xz exp(-x) - 2y^3 + y^2 - 3z^3
    
computed within a method that implements a generic fitness function interface 
    
    double fit(double p[])
    
with 

    x = p[0] 
    y = p[1]
    z = p[2]
    
all in the range `[0, 100]`

## Solution

I need to define the following entities

- Fitness function
- Individual
- Population
- Genetic operators
    - Selection
    - Crossover
    - Mutation

## Derivative

https://www.symbolab.com/solver/partial-derivative-calculator/%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%5Cleft(%5Cleft(2%5Ccdot%20x%5Ccdot%20z%5Ccdot%5Cleft(e%5E%7B-x%7D%5Cright)%20-%202%5Ccdot%20y%5E%7B3%7D%20%2B%20y%5E%7B2%7D%20-%203%5Ccdot%20z%5E%7B3%7D%5Cright)%5Cright)

$$\frac{\partial }{\partial x}\left(\left(2\cdot x\cdot z\cdot \left(e^{-x}\right)\:-\:2\cdot y^3\:+\:y^2\:-\:3\cdot z^3\right)\right)$$

https://www.wolframalpha.com/input/?i=maximize+2*x*z+*+exp%28-x%29+-+2*y%5E3+%2B+y%5E2+-+3*z%5E3