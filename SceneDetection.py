import numpy as np
import matplotlib.pyplot as plt

class SceneDetector:

    def __init__(self, width, height, numberOfFrames):
        self.frames = np.empty([numberOfFrames, height, width])
        self.framesCount = 0

    def captureFrame(self, frame):
        self.frames[self.framesCount] = frame
        self.framesCount +=1

    def getSceneChanges(self, threshold):

        avgDiffenences = np.empty([self.framesCount - 1])

        for i in range(1, self.framesCount - 1):
            avgDiffenences[i - 1] = np.average(np.abs(self.frames[i] - self.frames[i - 1]))

        sceneChanges = [0]
        for i in range(len(avgDiffenences)):
            if avgDiffenences[i] > threshold:
                sceneChanges.append(i)

        plt.stem(avgDiffenences)
        plt.show()

        return sceneChanges


