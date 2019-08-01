# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from datetime import date, time, datetime
from enum import Enum
import time
import db_work

# Извлечение 24 картинки

class CentroidTracker():
    def __init__(self, maxDisappeared):
        # db.outup()
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # new
        self.detected_objects = list()
        # self.detected_objects = list(())

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        # new
        # self.detected_objects.append(
        #    detectObject(self.objects[self.nextObjectID], self.nextObjectID, datetime.datetime.today()))
        self.detected_objects.append(DetectionObject(self.nextObjectID, centroid))
        # ....old....
        self.nextObjectID += 1

    def deregister(self, objectID):
        centroid = self.objects[objectID]
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        # obj = self.find_detectObject_by_ID(objectID)
        # obj.changeEndTime(datetime.datetime.today())
        del self.objects[objectID]
        del self.disappeared[objectID]
        # del list[object]
        # del self.objects[objectID]

        # new

        obj = self.find_detectObject_by_ID(objectID)
        if obj is not None:
            obj.delete(centroid)
            #del self.detected_objects[objectID]
        else:
            print("Object not found, id: " + objectID.__str__())

    def update(self, rects, images, ps):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared

            dis_cur = list(())

            for objectID in self.disappeared.keys():

                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    # Ломаецо
                    dis_cur.append(objectID)
                    # self.deregister(objectID)
            for index in dis_cur:
                self.deregister(index)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # new
                ob = self.find_detectObject_by_ID(objectID)
                if ob is not None:
                    ob.input_new_image(images[col], ps[col])
                else:
                    print("Something went wrong. CT can't find cur object, ID: " + objectID.__str__())

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects

    def find_detectObject_by_ID(self, id):
        for object in self.detected_objects:
            if object.id is id:
                return object
        return None

    def endCentroind(self):
        y = 10


class ObjectType(Enum):
    Unknown = 1
    Input = 2
    Output = 3


class Image_p:
    def __init__(self, image, p):
        self.image = image
        self.p = p


class DetectionObject:
    def __init__(self, object_id, begin_centroid):
        self.id = object_id
        self.begin_centroid = begin_centroid
        self.begin_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M')
        self.end_centroid = None
        self.end_time = None
        self.type = ObjectType.Unknown
        # Хранить не массив а одну картинку, при этом заменяя если вероятность выше!
        self.im_p = None
        print("Registered: " + object_id.__str__())

    def input_new_image(self, image, p):
        if self.im_p is not None:
            if self.im_p.p < p:
                self.im_p = Image_p(image, p)
        else:
            self.im_p = Image_p(image, p)

    def define_type(self):
        # Looking for centroid Y-axis
        d = self.end_centroid[1] - self.begin_centroid[1]
        if d > 0:
            self.type = ObjectType.Input
        elif d < 0:
            self.type = ObjectType.Output

    def base_token(self, begin_time, end_time, object_type, image):
        if image is not None:
            db_work.add_BD(begin_time, end_time, image, object_type)
            print("Запрос отправлен в БД")
        else:
            print("Плохое изображение")
        # Запрос в бд
        '''
        print("\n________________________________________________\n")
        print("Запрос в БД:")
        print("\nID: " + self.id.__str__())
        print("\nНачало: " + time.strftime("Year: %Y, Month: %m, Day: %d, Time: %H:%M:%S", begin_time))
        print("\nКонец: " + time.strftime("Year: %Y, Month: %m, Day: %d, Time: %H:%M:%S", end_time))
        print("\nТип: " + object_type)
        if image is None:
            print("\nИЗОБРАЖЕНИЕ ПОВРЕЖДЕНО")
        print("\n________________________________________________\n")
        if image is not None:
            import cv2
            cv2.imwrite('jpgs//' + self.id.__str__() + '.jpg', image)
        '''
        '''
        print("regist")
        if frame is not None:
            (startX, startY, endX, endY) = rects[i]
            sub_f = frame[startY:endY, startX:endX]
            cur_time = time.strftime("Year: %Y, Month: %m, Day: %d, Time: %H:%M:%S", time.localtime())
            #a = imwrite('jpegs//' + time.strftime("%H_%M_%S_", time.localtime()) + '.jpg', sub_f)
            imwrite('tst_to_db_jpg.jpg', sub_f)
            a = imread('tst_to_db_jpg.jpg')
            with open('tst_to_db_jpg.jpg', "rb") as f:
                contents = f.read().decode("UTF-8")
            #img_read = g_open.read()
            db.add_BD(cur_time, contents)
            print(cur_time)
        '''

    def delete(self, end_centroid):
        self.end_centroid = end_centroid
        self.end_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M')
        self.define_type()
        im = None
        if self.im_p is not None:
            im = self.im_p.image
        self.base_token(self.begin_time, self.end_time,
                        self.type.name.__str__(), im)
