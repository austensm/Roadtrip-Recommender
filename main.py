import sys
from src.utils import read_data, construct_categories_from_DF
from src.evaluate import KFoldCrossEvaluationForDecisionTree, KFoldCrossEvaluationForBackPropagation

def main():
    # Decision Tree Evaluation
    dtDF = read_data("data.txt")
    categorizedDf = construct_categories_from_DF(dtDF)

    attributeList = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]
    allLabels = ['very low', 'low', 'ok', 'high', 'great']

    print("Running Decision Tree Evaluation...")
    KFoldCrossEvaluationForDecisionTree(5, 1, categorizedDf, attributeList, allLabels, 5)

    print("\n\n")

    # Back Propagation Evaluation
    bpDF = read_data("data.txt")

    print("Running Back Propagation Evaluation...")
    KFoldCrossEvaluationForBackPropagation(5, 5, bpDF, 8)

if __name__ == "__main__":
    main()