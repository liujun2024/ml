""" 回归算法集成 """

from ._linearRegression import Linear
from ._decisionTree import DecisionTree, DT
# from ._gaussian import GaussR
from ._randomForest import RandomForest, RF
# from ._deepForest import DeepForest
from ._lightGradientBoostingMachine import LightGradientBoostingMachine, LightGBM
# from ._gradientBoostingDecisionTree import GradientBoostingDecisionTree, HistGradientBoostingDecisionTree
from ._extremeGradientBoosting import ExtremeGradientBoosting, XGBoost
# from ._neuralNetwork import NeuralNetwork

# from ._randomForest import suffix_kw