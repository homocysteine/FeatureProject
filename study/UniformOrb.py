import cv2
import numpy as np
from collections import deque

# QuadTree Filter
# We need a class represent Grid
# Grid: coordinate, flag represent if the node is valid
class Grid:
    def __init__(self, noMore=False, ulx=None, uly=None, brx=None, bry=None, feature=None,
                 end=False, next = None, prev = None):
        # if the node is valid?
        self.noMore = noMore
        # The coordinate of the grid
        self.ulx = ulx
        self.uly = uly
        self.brx = brx
        self.bry = bry
        # The KeyPoint Collection
        self.feature = feature
        # self.desc = desc
        # How many points the grid contains
        # As an end mark
        self.end = end
        self.next = next
        self.prev = prev

    def setKeyPoint(self,feature):
        self.feature = feature

    def divideNode(self):
        halfX = (self.brx-self.ulx) // 2
        halfY = (self.bry-self.uly) // 2
        # define children boundaries
        n1 = Grid(ulx=self.ulx,uly=self.uly,brx=self.ulx+halfX,bry=self.bry+halfY,feature=[])
        n2 = Grid(ulx=self.ulx+halfX,uly=self.uly, brx=self.brx, bry=self.uly+halfY,feature=[])
        n3 = Grid(ulx=self.ulx,uly=self.uly+halfY,brx=self.ulx+halfX,bry=self.bry,feature=[])
        n4 = Grid(ulx=self.ulx+halfX,uly=self.uly+halfY,brx=self.brx,bry=self.bry,feature=[])
        # associate children with their points
        for feat in self.feature:
            if feat.kp.pt[0] < n1.brx:
                if feat.kp.pt[1] < n1.bry:
                    n1.feature.append(feat)
                else:
                    n3.feature.append(feat)
            elif feat.kp.pt[1]<n1.bry:
                n2.feature.append(feat)
            else:
                n4.feature.append(feat)
        # set nomore flag
        if len(n1.feature) == 1:
            n1.noMore = True
        if len(n2.feature) == 1:
            n2.noMore = True
        if len(n3.feature) == 1:
            n3.noMore = True
        if len(n4.feature) == 1:
            n4.noMore = True
        # return 4 nodes
        return n1,n2,n3,n4

class Pair():
    def __init__(self,first=None, second=None):
        self.first = first
        self.second = second

# encapsulate the keypoint and descriptor
class Feature():
    def __init__(self,kp,desc):
        self.kp = kp
        self.desc = desc

def normalORB(image=None):
    orb = cv2.ORB_create(nfeatures=500)
    kp, desc = orb.detectAndCompute(image,None)
    # img = cv2.drawKeypoints(image,kp,np.array([]))
    # cv2.imshow('img',img)
    return kp


def quadTreeTest(method, nfeatures=500, image=None):
    # when feature points number is not enough, consider to adjust threshold or patch-size
    # Get the image size
    print(image.shape) # H, W
    height, width = image.shape
    # set the cell size
    w = 200
    # st the number of grid
    ncols = height // w
    nrows = width // w
    print(ncols, nrows)
    # calculate the cell size
    cellh = height // ncols
    cellw = width // nrows
    print(cellh, cellw)
    # Get points from each grid
    # Calculate the four points of grid
    # ncols = 41
    # nrows = 12

    # Grid detect
    kps = []
    gridList = [] # save the grid in the list
    flag = False # represent the grid has 0 or 1 keypoint.
    for i in range(ncols * nrows):
        nx = i % nrows # the index of the cell in x direction
        ny = i // nrows # the index of the cell in y direction
        ulx, brx = cellw * nx, cellw * (nx + 1)
        uly, bry = cellh * ny, cellh * (ny + 1)
        cropImg = image[uly:bry, ulx:brx]
        kp, desc = method.detectAndCompute(cropImg, None)
        featureList = []
        if len(kp) > 0:
            # print('find feature points!')
            # transform the relative position to absolute position
            # for k in kp:
            #     k.pt = (k.pt[0]+ulx, k.pt[1]+uly)
            for i in range(len(kp)):
                kp[i].pt = (kp[i].pt[0]+ulx,kp[i].pt[1]+uly)
                featureList.append(Feature(kp[i],desc[i]))
        grid = Grid(ulx=ulx, uly=uly, brx=brx, bry=bry, feature=featureList)
        gridList.append(grid)
        kps = kps + kp
    kpImg = cv2.drawKeypoints(image, kps, np.array([]))
    # cv2.imshow('kpImg', kpImg)
    print(len(kps))

    # mark the special node, the nodes with 1 or zero nodes
    def mark(lit):
        if len(lit.feature) == 1:
            lit.noMore = True
            return True
        elif len(lit.feature) == 0:
            return False
        else:
            return True

    res = filter(mark,gridList)
    # Here, GridList contains
    gridList = list(res)
    # End Signal
    gridList.append(Grid(end=True))

    # LinkList constructing(Bidirection linklist)
    for i in range(len(gridList)):# except the last one
        if i == 0:
            gridList[i].next = gridList[i+1]
        elif i == len(gridList)-1:
            gridList[i].prev = gridList[i-1]
        else:
            gridList[i].next = gridList[i+1]
            gridList[i].prev = gridList[i-1]

    bFinished = False
    vSizeAndPointerToNode = []
    while bFinished == False:
        preSize = len(gridList)
        lit = gridList[0] # point to the first node in gridList
        nToExpand = 0
        vSizeAndPointerToNode.clear()
        while lit.end != True:
            # contains only one point
            if lit.noMore:
                lit = lit.next
                continue
            # contains more than one points
            else:
                n1, n2, n3, n4 = lit.divideNode()
                # remove the current point,lit point to the next
                # remove the relationship in linklist
                if lit.prev != None:
                    lit.prev.next = lit.next
                    lit.next.prev = lit.prev
                else:
                    lit.next.prev = None
                temp = lit
                lit = lit.next
                gridList.remove(temp)
                # remove the relationship in list
                # gridList.pop(0)
                if len(n1.feature)>0:
                    n1.next = gridList[0]
                    gridList[0].prev = n1
                    gridList.insert(0, n1)
                    if len(n1.feature)>1:
                        nToExpand = nToExpand + 1
                        vSizeAndPointerToNode.append(Pair(len(n1.feature), gridList[0]))
                if len(n2.feature)>0:
                    n2.next = gridList[0]
                    gridList[0].prev = n2
                    gridList.insert(0,n2)
                    if len(n2.feature)>1:
                        nToExpand = nToExpand + 1
                        vSizeAndPointerToNode.append(Pair(len(n2.feature), gridList[0]))
                if len(n3.feature)>0:
                    n3.next = gridList[0]
                    gridList[0].prev = n3
                    gridList.insert(0,n3)
                    if len(n3.feature)>1:
                        nToExpand = nToExpand + 1
                        vSizeAndPointerToNode.append(Pair(len(n3.feature), gridList[0]))
                if len(n4.feature)>0:
                    n4.next = gridList[0]
                    gridList[0].prev = n4
                    gridList.insert(0,n4)
                    if len(n4.feature)>1:
                        nToExpand = nToExpand + 1
                        vSizeAndPointerToNode.append(Pair(len(n4.feature), gridList[0]))

        # judge if the cycle is end
        # condition1: The number of grid nodes is more than required points' number.
        # condition2: The quadtree cannot be separated again.(current list size is equal to the prev size)
        # list remove is delete based on value
        if(len(gridList)>=nfeatures or len(gridList)==preSize):
            bFinished = True
        elif (len(gridList)+ nToExpand*3)>nfeatures: # Is about to get the last split
            while(bFinished==False):# last split
                preSize = len(gridList)
                vPrevSizeAndPointerToNode = vSizeAndPointerToNode.copy() #return a copy
                vSizeAndPointerToNode.clear()
                vPrevSizeAndPointerToNode.sort(key=lambda pair:pair.first,reverse=True)
                # from more to less
                for j in range(len(vPrevSizeAndPointerToNode)):
                    n1, n2, n3, n4 = vPrevSizeAndPointerToNode[j].second.divideNode()
                    lit = vPrevSizeAndPointerToNode[j].second
                    if lit.prev != None:
                        lit.prev.next = lit.next
                        lit.next.prev = lit.prev
                    # lit = lit.next
                    gridList.remove(lit) #exist in vPre but not exist in gridList
                    if len(n1.feature) > 0:
                        n1.next = gridList[0]
                        gridList[0].prev = n1
                        gridList.insert(0, n1)
                        if len(n1.feature) > 1:
                            nToExpand = nToExpand + 1
                            vSizeAndPointerToNode.append(Pair(len(n1.feature), n1))
                    if len(n2.feature) > 0:
                        n2.next = gridList[0]
                        gridList[0].prev = n2
                        gridList.insert(0, n2)
                        if len(n2.feature) > 1:
                            nToExpand = nToExpand + 1
                            vSizeAndPointerToNode.append(Pair(len(n2.feature), n2))
                    if len(n3.feature) > 0:
                        n3.next = gridList[0]
                        gridList[0].prev = n3
                        gridList.insert(0, n3)
                        if len(n3.feature) > 1:
                            nToExpand = nToExpand + 1
                            vSizeAndPointerToNode.append(Pair(len(n3.feature), n3))
                    if len(n4.feature) > 0:
                        n4.next = gridList[0]
                        gridList[0].prev = n4
                        gridList.insert(0, n4)
                        if len(n4.feature) > 1:
                            nToExpand = nToExpand + 1
                            vSizeAndPointerToNode.append(Pair(len(n4.feature), n4))
                    if len(gridList)>= nfeatures:
                        break
                if len(gridList)>=nfeatures or len(gridList)==preSize:
                    bFinished = True
        # no else, because we will not process other conditions.

    # Retain the best point in each node
    kpList = []
    descList = []
    # for lit in gridList:
    for i in range(len(gridList)-1):
        maxp = gridList[i].feature[0]
        maxResponse = maxp.kp.response
        for j in range(len(gridList[i].feature)):
            if gridList[i].feature[j].kp.response > maxResponse:
                maxp = gridList[i].feature[j]
                maxResponse = maxp.kp.response
        kpList.append(maxp.kp)
        descList.append(maxp.desc)
    # descList->numpy
    descList = np.array(descList)
    return kpList,descList

    # cv2.waitKey(0)

if __name__ == '__main__':
    image_dir = '/media/yushichen/DATA/Datasets/data_odometry_gray/dataset/sequences/00/image_0/000000.png'
    image = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    # Create orb object
    orb = cv2.ORB_create(nfeatures=500)
    kp,desc = quadTreeTest(orb,nfeatures=500,image=image)
    # res = normalORB(image)
    kpImg = cv2.drawKeypoints(image, kp, np.array([]))
    cv2.imshow('kpImg',kpImg)
    cv2.waitKey(0)
    i = 0

