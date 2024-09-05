# Roadtrip Recomendation Program

## Project Information
- Class: CS-4260
- Professor: Douglas Fisher
- Contributers: Ronni Tong, Caroline Dunn, Sally Austen

## Intro

- Constructed and tested methods of machine learning on data about a user’s preferences for locations and for edges (for a Road Network). Single real values for utilities are computed from themes (aka features), each of which is assumed to have a micro-utility in the mind of a user. The function to compute an overall factor utility (or location and edge preference) over theme utilities will be learned from data on locations, edges, attractions, and themes.
- ML Algorithms implemented:
    - Decision Tree with variable maximum depth
    - Back propagation neural network with 1 hidden layer with variable size
- Evaluation of modles: k-fold evaluation--each parameter set with 5-fold evaluation

## Design

### Decision Tree
- **Gini splitting:** chosen for ease of implementation and good performance
    ![Alt text](context/Gini.png?raw=true "Equation of Gini Index")
- **Tree Structure**
    - Nodes either have 2 child nodes (feature node) or no child nodes (label node).
    - Tree built recursively by splitting datasets
- **Splitting Criteria**
    - Splits are made if there are still features to use or the tree hasn't reached its depth limit
    - Each feature's values are compared to a threshold (split number)
    - The split that results in the lowest Gini index (best separation) is chosen
- **Prediction Process**
    - Start at the root, fetch the feature (v) and split number (n)
    - Compare data entry's v value to n
    - Traverse left if v ≤ n, or right if v > n
    - Label data entry with the utility value of the leaf node reached

### Neural Network
- Used NumPy to create arrays for handling data and computations.
- Implemented forward and backward passes of the training process
- Experimented with different combinations of:
    - Epoch number (number of times the training data passes through the network)
    - Learning rate (how much the weights are updated during training)
    - Learning rate decay (reduces the learning rate over time to improve stability).
- Found the optimal combination of these parameters to achieve the best training performance


## Experimental Results
**Decision Tree**
    - When the tree has a max depth of 2, its accuracy is generally around 38% and variance around 0.001
    - When the tree has a max depth of 5, its accuracy is generally around 45% and variance around 0.001
    - The variance is similar for both depths, but with a bigger max depth, the tree’s accuracy increases. Still, it’s too low to be useful in production environments but pretty reasonable for the data volume and conditions.

**Back Propagation**
    - When the neural network has a hidden layer size of 4, its MSE is generally around 0.03 and variance around 1.9e-06
    - When the neural network has a hidden layer size of 8, its MSE is generally around 0.03 and variance around 6.3e-06
    - The MSE is similar for both depths, but with a bigger hidden layer size, the neural network’s variance increases