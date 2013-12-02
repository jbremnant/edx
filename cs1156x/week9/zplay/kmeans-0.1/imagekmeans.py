from PIL import Image
from toolbox.clustering.kmeans import KMeansCluster
from random import randrange
import math
import sys

class ImageKMeansCluster(KMeansCluster):
   def __init__(self, imageName, k):
      # set the file name and extension, to be used later when we output clustered image(s)
      self.fileName = imageName.split("/")[-1].split(".")[0]
      self.fileExtension = imageName.split(".")[1]

      self.clusterColors = [0x00b4ff, 0xffa61a, 0xff5e5e, 0x8ade8a, 0xbf6ec6]

      # set the number of dimensions
      self.k = k

      # open image and get size
      img = Image.open(imageName)
      self.width, self.height = img.size

      # get data points from image (in this case location and RGB color)
      self.data = []
      for row in range(self.height):
         for col in range(self.width):
            r,g,b = img.getpixel((col, row))
            self.data.append((col,row,r,g,b))

      # randomly generate centers (not ideal as centers could be equal)
      self.centers = []
      for i in range(k):
         x,y = randrange(self.height), randrange(self.width)
         r,g,b = randrange(256), randrange(256), randrange(256)
         self.centers.append([x,y,r,g,b])

   def saveClustersAsImages(self, clusters, bgcolor):
      "Saves each cluster as a distinct image with the specified background color"
      for i in range(len(clusters)):
         cImg = Image.new("RGB", (self.width,self.height), bgcolor)
         for (x,y,r,g,b) in clusters[i]:
            cImg.putpixel((x,y), (r,g,b))
         cImg.save(self.fileName + str(i+1) + "." + self.fileExtension)

   def saveClustersAsOneImage(self, clusters):
      cImg = Image.new("RGB", (self.width,self.height))
      for i in range(len(clusters)):
         for (x,y,r,g,b) in clusters[i]:
            cImg.putpixel((x,y), self.clusterColors[i])
      cImg.save(self.fileName + "_clustered." + self.fileExtension)

   def saveClusterColoredImage(self, clusters):
      cImg = Image.new("RGB", (self.width,self.height))
      for i in range(len(clusters)):
         avgColor = (int(self.centers[i][2]), int(self.centers[i][3]), int(self.centers[i][4]))
         for (x,y,r,g,b) in clusters[i]:
            cImg.putpixel((x,y), avgColor)
      cImg.save(self.fileName + "_clustered_" + str(self.k) + "." + self.fileExtension)

   def distance(self, pt1, pt2):
      totalDist = 0.
      for i in range(2,5):
         totalDist += (pt1[i] - pt2[i])**2
      totalDist = math.sqrt(totalDist)
      return totalDist

while True:
   try:
      im = ImageKMeansCluster(sys.argv[1], int(sys.argv[2]))
      clusters = im.getClusters()
      #im.saveClustersAsImages(clusters, 0xffffff)
      #im.saveClustersAsOneImage(clusters)
      im.saveClusterColoredImage(clusters)
      break
   except Exception:
      continue
