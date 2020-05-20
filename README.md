# README

## How to use this repo?

Close this repo, then I suggest you create a virtual environment and install the dependencies in `requirements.txt` with the following command

    pip install -r requirements.txt
    
Then just run `main.py`.

## Problem

Implement a genetic algorithm to maximize the numeric function 

    f(x, y, z) = 2*x*z * exp(-x) - 2*y^3 + y^2 - 3*z^3
    
such that `x`, `y` and `z` are in the range `[0, 100]`

## GA

To solve this problem with a GA, I need to define the following entities

- Fitness function
- Individual
- Population
- Genetic operators
    - Selection
    - Crossover
    - Mutation

See `main.py` for more details.

## Derivative

The local maximum of `f` can be found at the following URL

https://www.wolframalpha.com/input/?i=maximize+2*x*z+*+exp%28-x%29+-+2*y%5E3+%2B+y%5E2+-+3*z%5E3