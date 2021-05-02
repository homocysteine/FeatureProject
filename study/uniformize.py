import cv2
import numpy as np
from study.UniformOrb import Pair
class Feature():
    def __init__(self,kp,desc,scores):
        self.kp = kp
        self.desc = desc
        self.scores =scores

class Grid():
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
            if feat.kp[0] < n1.brx:
                if feat.kp[1] < n1.bry:
                    n1.feature.append(feat)
                else:
                    n3.feature.append(feat)
            elif feat.kp[1]<n1.bry:
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


def deepQuadTree(shape, kp, desc, scores,nfeatures=500):
    height, width = shape
    # set the cell size
    w = 150
    # st the number of grid
    ncols = height // w
    nrows = width // w
    print(ncols, nrows)
    # calculate the cell size
    cellh = height // ncols
    cellw = width // nrows
    print(cellh, cellw)
    # separate the keypoints to the correspondence grid
    gridList = []
    for i in range(ncols * nrows):
        nx = i % nrows # the index of the cell in x direction
        ny = i // nrows # the index of the cell in y direction
        ulx, brx = cellw * nx, cellw * (nx + 1)
        uly, bry = cellh * ny, cellh * (ny + 1)
        featureList=[]
        for j in range(len(kp)):
            if kp[j][0]<brx and kp[j][0]>=ulx and kp[j][1]<bry and kp[j][1]>=uly:
                featureList.append(Feature(kp=kp[j],desc=desc[j],scores=scores[j]))
        gridList.append(Grid(ulx=ulx, uly=uly, brx=brx, bry=bry, feature=featureList))
    # same workflow to the normal quadTree
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
        lit = gridList[0]  # point to the first node in gridList
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
                if len(n1.feature) > 0:
                    n1.next = gridList[0]
                    gridList[0].prev = n1
                    gridList.insert(0, n1)
                    if len(n1.feature) > 1:
                        nToExpand = nToExpand + 1
                        vSizeAndPointerToNode.append(Pair(len(n1.feature), gridList[0]))
                if len(n2.feature) > 0:
                    n2.next = gridList[0]
                    gridList[0].prev = n2
                    gridList.insert(0, n2)
                    if len(n2.feature) > 1:
                        nToExpand = nToExpand + 1
                        vSizeAndPointerToNode.append(Pair(len(n2.feature), gridList[0]))
                if len(n3.feature) > 0:
                    n3.next = gridList[0]
                    gridList[0].prev = n3
                    gridList.insert(0, n3)
                    if len(n3.feature) > 1:
                        nToExpand = nToExpand + 1
                        vSizeAndPointerToNode.append(Pair(len(n3.feature), gridList[0]))
                if len(n4.feature) > 0:
                    n4.next = gridList[0]
                    gridList[0].prev = n4
                    gridList.insert(0, n4)
                    if len(n4.feature) > 1:
                        nToExpand = nToExpand + 1
                        vSizeAndPointerToNode.append(Pair(len(n4.feature), gridList[0]))

        # judge if the cycle is end
        # condition1: The number of grid nodes is more than required points' number.
        # condition2: The quadtree cannot be separated again.(current list size is equal to the prev size)
        # list remove is delete based on value
        if (len(gridList) >= nfeatures or len(gridList) == preSize):
            bFinished = True
        elif (len(gridList) + nToExpand * 3) > nfeatures:  # Is about to get the last split
            while (bFinished == False):  # last split
                preSize = len(gridList)
                vPrevSizeAndPointerToNode = vSizeAndPointerToNode.copy()  # return a copy
                vSizeAndPointerToNode.clear()
                vPrevSizeAndPointerToNode.sort(key=lambda pair: pair.first, reverse=True)
                # from more to less
                for j in range(len(vPrevSizeAndPointerToNode)):
                    n1, n2, n3, n4 = vPrevSizeAndPointerToNode[j].second.divideNode()
                    lit = vPrevSizeAndPointerToNode[j].second
                    if lit.prev != None:
                        lit.prev.next = lit.next
                        lit.next.prev = lit.prev
                    # lit = lit.next
                    gridList.remove(lit)  # exist in vPre but not exist in gridList
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
                    if len(gridList) >= nfeatures:
                        break
                if len(gridList) >= nfeatures or len(gridList) == preSize:
                    bFinished = True
        # no else, because we will not process other conditions.

    # Retain the best point in each node
    kpList = []
    descList = []
    scoresList = []
    # for lit in gridList:
    for i in range(len(gridList) - 1):
        maxp = gridList[i].feature[0]
        maxResponse = maxp.scores
        for j in range(len(gridList[i].feature)):
            if gridList[i].feature[j].scores > maxResponse:
                maxp = gridList[i].feature[j]
                maxResponse = maxp.scores
        kpList.append(maxp.kp)
        descList.append(maxp.desc)
        scoresList.append(maxp.scores)
    # descList->numpy
    kpList = np.array(kpList)
    descList = np.array(descList)
    scoresList = np.array(scoresList)
    return kpList, descList, scoresList