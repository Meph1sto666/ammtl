import cv2
import cv2.typing

def distance(c1:list[cv2.typing.MatLike], c2:list[cv2.typing.MatLike]) -> float:
	sumX1:int = 0
	sumX2:int = 0
	sumY1:int = 0
	sumY2:int = 0
	for c1s in c1:
		M1:cv2.typing.Moments = cv2.moments(c1s)
		cX1:int = int(M1['m10'] / M1['m00'])
		cY1:int = int(M1['m01'] / M1['m00'])
		sumX1+=cX1
		sumY1+=cY1
	for c2s in c1:
		M2:cv2.typing.Moments = cv2.moments(c2s)
		cX2:int = int(M2['m10'] / M2['m00'])
		cY2:int = int(M2['m01'] / M2['m00'])
		sumX2+=cX2
		sumY2+=cY2
	avgX1=int(sumX1/len(c1))
	avgY1=int(sumY1/len(c1))
	avgX2=int(sumX2/len(c2))
	avgY2=int(sumY2/len(c2))
	return ((avgX2 - avgX1)**2 + (avgY2 - avgY1)**2)**0.5

def cluster(contours:list[cv2.typing.MatLike]) -> list[list[cv2.typing.MatLike]]:
	clusters:list[list[cv2.typing.MatLike]] = []
	for c in contours:
		clusters.append([c])
	popped = []
	for i in range(len(clusters)):
		for j in range(i+1, len(clusters)):
			d:float = distance(clusters[i],clusters[j])
			if i in popped: continue
			if d > 100: continue
			clusters[i].extend(clusters[j])
			popped.append(j)
	# print(clusters)
	return clusters

# def mergeCluster()