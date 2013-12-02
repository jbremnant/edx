from toolbox.clustering.kmeans import KMeansCluster
from random import randrange
import math

class NumbersKMeansCluster(KMeansCluster):
   def __init__(self, data, k):
      self.k = k
      self.data = data
      self.centers = [0 for i in range(k)]
      minVal = min(data)
      maxVal = max(data)
      for i in range(k):
         self.centers[i] = randrange(minVal, maxVal)
         while i != 0 and self.centers[i] in self.centers[0:i-1]:
            self.centers[i] = randrange(minVal, maxVal)
      print self.centers

   def distance(self, pt1, pt2):
      return abs(pt1-pt2)

   def getNewCenters(self,clusters):
      tmpCenters = []
      for i in range(len(clusters)):
         tmpCenter = 0.
         for val in clusters[i]:
            tmpCenter += val
         tmpCenter /= len(clusters[i])
         tmpCenters.append(tmpCenter)
      return tmpCenters

data = [1,3,4,12,52,22,11,12,13,567,743,423,456,8,45,60,2,11]
nc = NumbersKMeansCluster(data, 2)
clusters = nc.getClusters()
for c in clusters:
   print c
