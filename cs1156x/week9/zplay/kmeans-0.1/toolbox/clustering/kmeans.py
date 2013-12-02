"""
   kmeans.py

   Clusters a set of data points using Lloyd's algorithm, as follows:
      - Given k initial cluster centers and a set of data points
      - Repeat
         - Assign each point to one of k clusters by distance from cluster centers,
           adding to variance each time
         - if variance is less than threshold, reset the centers and return the clusters

   Copyright (c) 2008 Jordan Timmermann. All rights reserved.
"""
import math
import sys

class KMeansCluster: 
   def __init__(self):
      "The following variables need to be set in order to run the clusterer"
      self.data = [] # data points to be clustered
      self.k = 0 # number of clusters
      self.centers = [] # initial cluster centers

   def getClusters(self):
      "Returns clusters of data points"
      # initialize the last observed variance in data points from their cluster centers
      prevVariance = None

      ctr = 1
      # continue reclustering points until variance converges to zero
      while True:
         # reset the variance and clusters
         curVariance = 0.
         clusters = [[] for i in range(self.k)]
         # create new clusters based on current cluster centers
         for point in self.data:
           # assign point to cluster with nearest center
           clusterIndex, dist = self.getBestCluster(point)
           clusters[clusterIndex].append(point)
           
           # increase the variance by the distance from the data point to its cluster's center
           curVariance += dist
         
         if prevVariance != None:
            sys.stdout.write("Iteration " + str(ctr) + "\r")
            sys.stdout.flush()
            ctr += 1

         # if the variance has not changed since the previous iteration, return the clusters
         if prevVariance != None and (prevVariance - curVariance == 0.):
            return clusters
        
         # store the new variance as the previous variance, for next iteration
         prevVariance = curVariance

         self.centers = self.getNewCenters(clusters)

   def getNewCenters(self, clusters):
      "Returns centroids of clusters; assumes all dimensions of data points should be used"
      # use the first data point to determine dimensionality of data points
      dims = len(self.data[0])
      centers = []
      for ctr in range(self.k):
         cluster = clusters[ctr]
         tmpCenter = [0. for i in range(dims)]
         for point in cluster:
            for i in range(dims):
               tmpCenter[i] += point[i]
         for i in range(dims):
            tmpCenter[i] /= len(cluster)
         centers.append(tmpCenter)
      return centers

   def getBestCluster(self, point):
      "Given a data point, returns the index of the cluster whose center it is closest to,  \
      along with the distance to that center"
      # initialize minDist to be infinity
      minDist = ()
      # find minimum distance over all cluster centers
      for i in range(self.k):
         dist = self.distance(point, self.centers[i])
         if dist < minDist:
            minDist = dist
            bestCluster = i
      return bestCluster, minDist

   def distance(self, pt1, pt2):
      "Default distance measure; uses Euclidean distance"
      totalDist = 0.
      # use every dimension in the data point
      for i in range(len(pt1)):
         totalDist += (pt1[i] - pt2[i])**2
      totalDist = math.sqrt(totalDist)
      return totalDist
