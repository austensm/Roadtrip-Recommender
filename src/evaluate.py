import numpy as np
import pandas as pd
from copy import deepcopy

from src.decision_tree import DecisionTree
from src.node import Node
from back_propogation import BackPropagation

def KFoldCrossEvaluationForDecisionTree(k, totalRunCount, df, attributeList, allLabels, maxDepth):
  finalAccuracy = 0
  finalVariance = 0
  allFinalAccuracyList = []
  print("Decision tree stats:")

  tmpDT = DecisionTree(maxDepth)

  # go through each run and perform k-fold evaluation
  for currentRunNum in range(totalRunCount):
    accuracy = 0
    variance = 0
    allAccuracyList = []
    print("Current Run:", currentRunNum + 1)

    # divide into k folds
    shuffled = df.sample(frac = 1)
    result = np.array_split(shuffled, k)
    for i in range(k):
      # use ith fold as test data, remaining folds as training data
      testData = result[i]
      concatList = []
      for j in range(k):
        if j != i:
          concatList.append(result[j])
      trainingData = pd.concat(concatList)

      # train model with training Data
      model = deepcopy(tmpDT)
      model.root = Node()
      print(model.root.attribute)
      model.build_tree(deepcopy(trainingData), attributeList, allLabels, 0, model.root)
      prediction = model.predict(deepcopy(testData))

      # find accuracy and error
      # accuracy is the percentage of correctness for prediction
      # variance = SUM(prediction[i] - target[i])^2 / fold size
      correctCount = 0
      currentVariance = 0
      for j in range(len(prediction)):
        if prediction[j] == testData.iloc[j]["utility"]:
          correctCount += 1

      currentAccuracy = correctCount / len(prediction)
      currentVariance /= len(prediction)

      print("Current fold:", i + 1, "Current Accuracy:", currentAccuracy)

      allAccuracyList.append(currentAccuracy)
      accuracy += currentAccuracy

    # calculate variance
    accuracy /= k
    for i in allAccuracyList:
      variance += pow(i - accuracy, 2)
    variance /= (k - 1)

    print("Total Accuracy:", accuracy, "Total variance:", variance)

    # Add current run stats to final stats
    finalAccuracy += accuracy
    allFinalAccuracyList.extend(allAccuracyList)

    # blank line to separate runs
    print()


  # Calculate final accuracy and variance
  # by pooling together all accuracies
  print("Final Decision tree stats:")
  finalAccuracy /= totalRunCount
  for i in allFinalAccuracyList:
    finalVariance += pow(i - finalAccuracy, 2)
  finalVariance /= (k * totalRunCount - 1)
  print("Final Accuracy:", finalAccuracy, "Final variance:", finalVariance)
  return

def KFoldCrossEvaluationForBackPropagation(k, totalRunCount, df, hiddenSize):
  finalMSE = 0
  finalVariance = 0
  allFinalMSEList = []
  print("Back propagation stats:")

  tmpBP = BackPropagation(10, 1, hiddenSize)

  # go through each run and perform k-fold evaluation
  for currentRunNum in range(totalRunCount):

    allMSEList = []
    mse = 0
    variance = 0
    print("Current Run:", currentRunNum + 1)

    # divide into k folds
    shuffled = deepcopy(df).sample(frac = 1)
    result = np.array_split(shuffled, k)
    for i in range(k):
      bp = deepcopy(tmpBP)

      # use ith fold as test data, remaining folds as training data
      testData = result[i]
      concatList = []
      for j in range(k):
        if j != i:
          concatList.append(result[j])
      trainingData = pd.concat(concatList)

      trainingDataAttributes = trainingData.drop(columns = ["utility"]).to_numpy()
      trainingDataTargets = trainingData["utility"].to_numpy()
      testDataAttributes = result[1].drop(columns = ["utility"]).to_numpy()
      testDataTargets = result[1]["utility"].to_numpy()

      # train model with training Data
      bp.learn(trainingDataAttributes, trainingDataTargets, 100, 0.0001, 0.032)
      prediction = bp.predict(testDataAttributes)

      # find accuracy and error
      # accuracy is the percentage of correctness for prediction
      # variance = SUM(prediction[i] - target[i])^2 / fold size
      currentMSE = 0
      currentVariance = 0
      for j in range(len(prediction)):
        #print(prediction[j][0], testDataTargets[j])
        currentMSE += pow((prediction[j] - testDataTargets[j]), 2)
      currentMSE = currentMSE / len(prediction)
      print("Current fold:", i + 1, "Current MSE:", currentMSE)
      allMSEList.append(currentMSE)
      mse += currentMSE

    # calculate variance
    mse /= k
    for i in allMSEList:
      variance += pow(i - mse, 2)
    variance /= (k - 1)

    print("Total MSE:", mse, "Total variance:", variance)

    # Add current run stats to final stats
    finalMSE += mse
    allFinalMSEList.extend(allMSEList)

    # blank line to separate runs
    print()


  # Calculate final accuracy and variance
  # by pooling together all mses
  print("Final Back Propagation stats:")
  finalMSE /= totalRunCount
  for i in allFinalMSEList:
    finalVariance += pow(i - finalMSE, 2)
  finalVariance /= (k * totalRunCount - 1)
  print("Final MSE:", finalMSE, "Final variance:", finalVariance)
  return finalMSE, finalVariance