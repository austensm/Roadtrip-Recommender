import pandas as pd
from copy import deepcopy

from src.node import Node

class DecisionTree:
  def __init__(self, maxDepth = 5):
    self.root = None
    self.maxDepth = maxDepth


  def getSplitGini(self, firstPart, secondPart, trainingData, allLabels):
    if len(trainingData) == 0:
      return 0
    if len(firstPart) == 0 or len(secondPart) == 0:
      return 0
    # calculate split score for first part
    score = 0
    if len(firstPart) == 0:
      return 0
    for i in allLabels:
      pi = pow(len(firstPart.loc[firstPart["utility"] == i]) / len(firstPart), 2)
      score += pi

    if len(secondPart) == 0:
      return 0
    for i in range(11):
      pi = pow(len(secondPart.loc[secondPart["utility"] == i]) / len(secondPart), 2)
      score += pi
    return score


  def getSplitNum(self, trainingData, currentAttribute, allLabels):
    # go through all possible splits for the attribute
    # and find one with best division of labels, i.e. lowest entropy
    maxGini = 0
    splitNum = 0
    for i in range(0, 11):
      firstSplit = pd.DataFrame()
      secondSplit = pd.DataFrame()
      firstSplit = trainingData.loc[trainingData[currentAttribute] <= i]
      secondSplit = trainingData.loc[trainingData[currentAttribute] > i]
      currentSplitGini = self.getSplitGini(firstSplit, secondSplit, trainingData, allLabels)
      if currentSplitGini > maxGini:
        maxGini = currentSplitGini
        splitNum = i
    return splitNum, maxGini

  # build the decision tree
  def build_tree(self, trainingData, attributeCandidates, allLabels, depth = 0, currentNode = Node()):
    # for current layer, if all attributes are used, set leave with a label
    if len(attributeCandidates) == 0 or depth == self.maxDepth:
      currentNode.label = trainingData.mode().iloc[0]["utility"]
      currentNode.children = []
      return


    # find best attribute
    maxGini = 0
    splitNum = 0
    bestAttribute = ""
    for currentAttribute in attributeCandidates:
      currentSplitNum, currentGini = self.getSplitNum(trainingData, currentAttribute, allLabels)
      if currentGini >= maxGini:
        maxGini = currentGini
        bestAttribute = currentAttribute
        splitNum = currentSplitNum



    firstSplit = trainingData.loc[trainingData[bestAttribute] <= splitNum]
    secondSplit = trainingData.loc[trainingData[bestAttribute] > splitNum]
    currentNode.leftDF = firstSplit
    currentNode.rightDF = secondSplit

    if len(firstSplit) == 0 or len(secondSplit) == 0:
      currentNode.label = trainingData.mode().iloc[0]["utility"]
      currentNode.children = []
      return

    # go to left and right subtree
    attributeCandidates.remove(bestAttribute)
    currentNode.attribute = bestAttribute
    currentNode.splitNum = splitNum
    currentNode.children = [Node(), Node()]
    tmpAttributeCandidate = deepcopy(attributeCandidates)

    # build left subtree
    self.build_tree(firstSplit, tmpAttributeCandidate, allLabels, depth + 1, currentNode.children[0])
    # build right subtree
    tmpAttributeCandidate = deepcopy(attributeCandidates)
    self.build_tree(secondSplit, tmpAttributeCandidate, allLabels, depth + 1, currentNode.children[1])
    return



  def predict(self, trainingData):
    resultLabels = []
    for i in range(0, len(trainingData)):
      currentNode = self.root
      # traverse tree to find label
      while currentNode.children != []:
        if trainingData.iloc[i][currentNode.attribute] <= currentNode.splitNum:
          currentNode = currentNode.children[0]
        else:
          currentNode = currentNode.children[1]
      resultLabels.append(currentNode.label)
    return resultLabels

