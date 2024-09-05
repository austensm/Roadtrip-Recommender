import pandas as pd

# each node has left node and right node
# left splits has attribute value no bigger than split num
class Node:
  def __init__(self, children = [], splitNum = 0, attribute = ""):
    self.splitNum = splitNum
    self.children = children
    self.attribute = attribute
    self.label = None
    self.leftDF = pd.DataFrame()
    self.rightDF = pd.DataFrame()

  def set_label(self, newLabel):
    self.label = newLabel