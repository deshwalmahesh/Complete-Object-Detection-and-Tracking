import imutils
from imutils.video import VideoStream
import cv2
import glob
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import time
import warnings
import random

    
class BoundingBoxTracker():
    def __init__(self,dir_name:[None,bool,str]="tracking-data",ext:[None,bool,str]='jpg',tracker_type:str='KCF'):
        '''
        Class to detect the object based on bounding boxes. It uses different algorithms. 
        NNEDS A BOUNDING BOX FOR SURE IN tHE STARTING FOR THE OBJECTS YOU WANT TO TRACK.
        THIS CODE IS FOR DIRECTORY OF IMAGES. Change it for Videos or Live Camera too. Need a bit of changes only
        args:
            dir_name: directory name which has images
            ext: extension type of images
            tracker: names for tracker API
        '''
        self.files = sorted(glob.glob(dir_name+'/*.'+ext))
        
        tracker_type = tracker_type.upper()
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
            
        self.tracker =  tracker


    def track(self,initBB:[None,bool,tuple]=None,fps:int=20,resize:bool=False):
        '''
        Track an object in the sequence of images or Video
        args:
            initBB: initial Bounding box
            fps: frame per second. Not exactly the fpd but it means each frame willl be displayed for these many miliseconds
        returns:
            dict of frame numbers and if there was a bounding box of object found
        '''
        self.initBB = initBB
        self.fps = fps # 20 means show a frame for 20 miliseconds and then show next frame
        self.positions = {} # frame number and position of object (x,y,w,h)

        for index,i in enumerate(self.files):
            frame = cv2.imread(i)
            if resize:
                frame = imutils.resize(frame, width=500)
            (H, _) = frame.shape[:2] # get height and width


            if self.initBB is not None:
                (success, box) = self.tracker.update(frame)

                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
                    self.positions[index] = (x, y, w, h)

                info = [("Tracker", 'kcf'),("Success", "Yes" if success else "No"),]
                
                # loop over the info tuples and draw them on our frame
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


            cv2.imshow("Frame", frame)
            if not self.initBB:
                if index==0: # freeze the video at first frame and ask for ROI
                    self.initBB = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)
                    self.tracker.init(frame, self.initBB)
            else:
                self.tracker.init(frame, self.initBB)
                
            key = cv2.waitKey(self.fps) & 0xFF
            if key == ord("q"):
                break      

        cv2.destroyAllWindows() # else it'll just make the kernel crasj in Windows System


class ColorMaskBasedTracker():
    '''
    Class to detect Objects based on the basis of Color schemes. A single type of color is visible only by changing the bar that'll be created.
    Environment needs to be Vibrant and colorful to detect objects else it'll just show all the same color type objects together.
    '''

    def dummy(self,x=None):
        '''
        Method used as Dummy because in the CreateTrackbar, it is a must. 
        args:
            x: This does nnot mean anything. Without this, the code might print some statements but the code will still run smoothly
        '''
        ... # Google 'Ellipsis' for these 3 dots . Just another thing for 'pass'
    

    def track(self,filename:[str,None,bool]=None,crop:[tuple,list]=(480,480)):
        '''
        Track the Objects
        args:
            filename: filename for a valid Video file type
            crop: Crop the original video by how how much size
        '''
        cap = cv2.VideoCapture(0) if not filename else  cv2.VideoCapture(filename) # use file else use live Camera

        cv2.namedWindow('Tracking Window')
        cv2.createTrackbar('LH','Tracking Window',0,255,self.dummy) # Lower Hie
        cv2.createTrackbar('UH','Tracking Window',255,255,self.dummy) # Upper Hue. Upper bounds are always in 255
        cv2.createTrackbar('LS','Tracking Window',0,255,self.dummy) # Lower saturation
        cv2.createTrackbar('US','Tracking Window',255,255,self.dummy) # upper saturation
        cv2.createTrackbar('LV','Tracking Window',0,255,self.dummy) # lower value
        cv2.createTrackbar('UV','Tracking Window',255,255,self.dummy) # upper value

        while cap.isOpened(): # keep showing the Image
            _,frame = cap.read()

            if crop:
                frame = cv2.resize(frame, crop)
            hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

            LH = cv2.getTrackbarPos('LH','Tracking Window')
            UH = cv2.getTrackbarPos('UH','Tracking Window')
            LS = cv2.getTrackbarPos('LS','Tracking Window')
            US = cv2.getTrackbarPos('US','Tracking Window')
            LV = cv2.getTrackbarPos('LV','Tracking Window')
            UV = cv2.getTrackbarPos('UH','Tracking Window')

            lower_bound = np.array([LH,LS,LV])
            upper_bound = np.array([UH,US,UV])

            object_mask = cv2.inRange(hsv_frame,lower_bound,upper_bound) # makes everything black except the color values in this range
            resulting_img = cv2.bitwise_and(frame,frame,mask=object_mask)

            cv2.imshow('Resulting Img',resulting_img)
            key = cv2.waitKey(1) # show for 1 miliseconds. Because the loop is infinite, it'll be infinitely showing the results
            if key==27 or key == ord('q'): # Press escape / q  to close all windows
                break

        cap.release()
        cv2.destroyAllWindows()


class ContourBasedTracker():
    '''
    Works on Backgrounnd Segmentation of Images
    Class to track the objects which are a part of rule based Contours. Objects coming within those contours will be detected. 
    There many parameters which you need to tweak in order to get you results. Method is as follows:
    Grayscale -> Gauss Filter -> Thresholding -> Dialatio -> Find Contour -> Reule to keep/remove eligible contours -> Plot BB
    '''
    def track(self,filename:[str,None,bool]=None,crop:[tuple,list]=(480,480),
        ksize:[tuple,list]=(3,3),sigmaX:int=0,
        thresh:int=15,maxval:int=255,type=cv2.THRESH_BINARY,kernel:[None,int]=None,iter=5,
        cont_mode=cv2.RETR_TREE,cont_method=cv2.CHAIN_APPROX_SIMPLE,cont_area_filter:int=750):
        '''
        Track the Objects
        args:
            filename: filename for a valid Video file type
            crop: Crop the original video by how how much size
            ksize: Kernel size of Gaussian Blur
            sigmaX: Standard Deviation in X direction of Gaussian Blur
            thresh: threshold value to use in the binarization
            maxval: MAx value of images. Upper bound of pixel values is default 255 set
            type: type of thresholding to use for binarization
            kernel:  Kernel size for Dilation
            iter: No of iterations to run
            cont_mode: Mode to find the contour. Check list as OpenCv website to use others
            cont_method: Exact method to use for finding contours
            cont_area_filter: Area of contour filter. Area less than this won't count as a good finding
        '''
        cap = cv2.VideoCapture(0) if not filename else cv2.VideoCapture(filename) # these works like Generator

        _,curr_frame = cap.read() # read first frame. Starting point
        _,next_frame = cap.read() # second frame
        if crop:
            curr_frame = cv2.resize(curr_frame, crop)
            next_frame = cv2.resize(next_frame, crop)
            
        while cap.isOpened(): # while there is a frame remaining
            diff = cv2.absdiff(curr_frame,next_frame) # Get the ABSOLUTE difference of the two frames
            grey = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY) # convert to grayscale for smoothing and then finding contours
            smooth = cv2.GaussianBlur(grey,ksize=ksize,sigmaX=sigmaX) # change these params for your image specific
            _,binary = cv2.threshold(smooth,thresh=thresh,maxval=maxval,type=type) #these are parameters agaib for your video specific
            dilated = cv2.dilate(binary,kernel=kernel,iterations=iter) # dilate to fill the holes or small patches
            contours,_ = cv2.findContours(dilated,mode=cont_mode,method=cont_method) # find the contours
            
            for contour in contours: # get each contour and filter it based on the rule. Chose your rules based on objects you are trying to find
                if cv2.contourArea(contour) > cont_area_filter: # get the contour IFF it has an area greater than this area
                    (x,y,w,h) = cv2.boundingRect(contour) # get minimal shape/ coordinates of rectanle which fits the given countor
                    if h>w: # for people as they'll have height greater than width. Again , choose your rules
                        cv2.rectangle(curr_frame,pt1=(x,y),pt2=(x+w,y+h),color=(255,0,1),thickness=2) # draw a red box around the contour of thickness 3
                        cv2.putText(curr_frame,text="Analysis: Moving",org=(10,10),fontobject=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0),thickness=3)
                        # put a font of Hershey style at coordinates (10,10) of thickness 4 with Green Color which says: "Analysis: Moving"

            cv2.imshow('Video',curr_frame)
            curr_frame = next_frame
            _, next_frame = cap.read() # swapping
            if crop:
                next_frame = cv2.resize(next_frame, crop)

            key = cv2.waitKey(10) # change this parameter for slow or fast image visualizations
            if key==27 or key == ord('q'): # Press escape / q  to close all windows
                break

        cap.release()
        cv2.destroyAllWindows()


class TrackCentroid():
    '''
    Class to Track a CENTROId of a bounding box. Bounding boxes are provided by any algorithm such as YOLO, SVM, HaarCascade etc.
    Works on the basis that an image or a patch is a distribution having a median and the assumption that centroid does not moves all of a sudden in a frame or becomes a ghost. 
    With each frame, check all the centroids and get their distance with each other.The centroids which is closet in the next frame is the same (assumption).
    '''
    def __init__(self,disappear_limit:int=50):
        '''
        args:
            disappear_limit: If an object id has not been detected for continuous these many no of frames, remove its id
        '''
        self.disappear_limit = disappear_limit # threshold for deleting object id
        self.next_object_id = 1 # in the starting it is 0
        self.objects_dict = OrderedDict() # object id mapping to its centroid
        self.missing_from = OrderedDict() # object id which is missing from these many frames

    
    def add_new_object(self,centroid:[tuple,list,np.ndarray]):
        '''
        Adds a new object in the objects_dict
        args:
            centroid: NEW centroid of the distribution of the object that has been detected
        '''
        assert len(centroid) == 2 ,"Centroid should have X and Y values"
        self.objects_dict[self.next_object_id] = centroid # add a new object tothe dictonary. For first object, it is 0
        self.missing_from[self.next_object_id] = 0 # because it is new, it is missing from 0 frames
        self.next_object_id+=1 # this will be used as new id for new object encountered in the distribution


    def remove_object(self,object_id:int):
        '''
        If a object has been missing from previous THRESHOLD frames, remove it
        args:
            object_id: id of the object to be removed from the dictonary
        '''
        del self.objects_dict[object_id]
        del self.missing_from[object_id]


    def refresh(self,rectangles:(tuple,list))->dict:
        '''
        Check for every frame if there is a new entry, we have to add, remove or delete something
        args:
            rectangles: bounding boxes or rectangle coordinates (x_start,y_start,x_end,y_end) of the new objects. IF there are any, use logic and add remove or track those
        '''
        assert isinstance(rectangles,(np.ndarray,tuple,list)), "Input should be of type tuple, list or numpy array"
        if len(rectangles)==0: # first condition if there was not a single object in the frame
            for object_id in list(self.objects_dict.keys()): # get every object id and increase the counter because there are no objects in the frame
                self.missing_from[object_id]+=1
                if self.missing_from[object_id] > self.disappear_limit: # if its value has crossed the threshold, remove it
                    self.remove_object(object_id)
            return self.objects_dict # nothing new to update

        # if there are 1 or more bounding boxes returned by the feature extractor
        input_centroids = np.zeros((len(rectangles),2),dtype='int32') # make a 2D array of X,Y for EACH bounding predicted
        for i,value in enumerate(rectangles):
            x_start, y_start, x_end, y_end = value # tuple unpacking
            c_x = int((x_start + x_end)/2) # get the Center of Median of X
            c_y = int((y_start + y_end)/2) # centroid of y values

            input_centroids[i] = (c_x, c_y) # change thew values from 0 to the values calculated for each individual entry

        if len(self.objects_dict) == 0: # if we have just started the video for the first time OR there we re no objects for threshold frames
            for centroid in input_centroids: # we need to add every centroid to the dict
                self.add_new_object(centroid) # add new object
        
        else: # if we already some values in the dict, then try to find which centroid belongs which of the existing and which one are new
            object_ids = list(self.objects_dict.keys()) # get all the object ids
            old_centroids = list(self.objects_dict.values()) # get the corresponding centroids
            distances = dist.cdist(np.array(old_centroids), input_centroids) # Google "Similiarity Search". Compare each value with every other and find the smallest distance for each
            rows = distances.min(axis=1).argsort() # get the minimum distances for EACH and sort them NOTE: It will give you the indices not the actual values check numpy docs
            cols = distances.argmin(axis=1)[rows] # it'll sort the columns and then give to corresonding axes
            used_rows, used_cols = set(), set() # ids which are already present and we have update their NEW centroid

            for (row,col) in zip(rows,cols): # get each row which corresponds to its new col
                if row in used_rows or col in used_cols: # update only if have not updated before recently
                    continue

                object_id = object_ids[row] # line 273. Get the object id using the index
                self.objects_dict[object_id]  = input_centroids[col] # replace old centroid with the new one
                self.missing_from[object_id] = 0 # we found it again so set the counter to 0

                used_rows.add(row) # add to set so we don't have to come again to this block for the same id
                used_cols.add(col)
            
            unused_rows = set(range(0, distances.shape[0])).difference(used_rows) # make a set of remaing idswhich are un accounted for
            unused_cols = set(range(0, distances.shape[1])).difference(used_cols) # get all the new one

            if distances.shape[0] >= distances.shape[1]: # if the old existing centroids  are > new centroids 
                for row in unused_rows: # loop over every existing object to increase the cunted because this is the only case if some are are accounted for even when they are nto in frame
                    object_id = object_ids[row] # get the old id
                    self.missing_from[object_id] += 1 # it is missing that is why over 
                    
                    if self.missing_from[object_id] > self.disappear_limit: # if it has crossed, remove the object
                        self.remove_object(object_id)

            else: # if it is less, just add new objects
                for col in unused_cols:
                    self.add_new_object(input_centroids[col])

        return self.objects_dict


class DLObjectDetector():
    '''
    Classs to Extract features / Objects from an Image using any Deep Learning Model. Use it for different classes or models. Works perfectly with YOLO
    '''
    def __init__(self,config:str,model_weight:str,cuda:bool=False,size:[tuple,list]=(416,416)):
        '''
        args:
            config: path to config file
            weight: path of weight file
            cuda: whether to use GPU and CudNN algos
            size: size of the input image
        '''
        warnings.warn("'opencv-contrib' needed. Use 'pip install opencv-contrib'") 
        self.model_weight = model_weight
        self.config = config
        self.size = size
        net = cv2.dnn.readNet(self.model_weight,self.config) # Load weight and config file
        if cuda: # activate GPU algo
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.model = cv2.dnn_DetectionModel(net) # load the exact model
        self.model.setInputParams(size=self.size,scale=1/255)
        self.centroid_tracker = TrackCentroid()


    def get_all_features(self,frame,nms:float=0.45,confidence:float=0.51,):
        '''
        Get class, score and all the bounding boxes which qualifies the given criteria of nms and confidence
        args:
            frame: Input image or Frame of a video, Camera
            nms: Non max supression threshold. BB confidence less than this value will be ignored
            conf: Confidence that an object has been found. Objects with values less than this will be ignored
        out: 
            Tuple of list of Valid BB boxes: All classes, All score, All Boxes, inference time: ([c1,c2,cw],[s1,s2,s3],[b1,b2,b3])
            NOTE: b1,b2,b3 are themselves in the format as b1 = [x1,y1,w1,h1], b2 = [x2,y2,w2,h2] 
        '''
        start = time.time()
        classes, scores, boxes = self.model.detect(frame, confidence, nms) # get the array of classes,probs for each class and BB
        end = time.time()
        return classes, scores, boxes, end-start

    
    def perform(self,classes_file:str,filepath:[str,bool,None]=None,kind:str='detect',resize:bool=False,nms:float=0.45,confidence:float=0.51,
        find:[None,list,tuple]=None,disappear_limit:int=50,count_at_t:bool=True,unique:bool=True):
        '''
        Get features from a DNN model in a Video and display it in real time
        args:
            classes_file: classes file names
            filepath: Path of the video file. If not provided, Use the Video Camera
            kind: Detection or tracking
            size: size of the input image
            resize: Whether to resize the image or use the original
            nms: Non max supression threshold. BB confidence less than this value will be ignored
            conf: Confidence that an object has been found. Objects with values less than this will be ignored
            find: Classes to look for. Classes should be in in 'classes' file. None returns all the classes
            count_at_t: Whether to count the number of objects in a certain time t . Valid for Tracking Only
            unique: Whether to point the total number of unique objects encountered up until now. Valid for Tracking Only
            disappear_limit: remove the object id if it isn't in the frame for these many frames. Valid for Tracking Only
        '''
        cap = cv2.VideoCapture(0) if not filepath else cv2.VideoCapture(filepath)

        with open(classes_file, "r") as f: # open the class name file
            class_dict = {i:cname.strip() for i,cname in enumerate(f.readlines())} # make dictonary of class name with the index
            class_names = list(class_dict.values())

        COLORS = np.random.randint(0,256,size=(len(class_names),3)).tolist() # generate different color for different classes

        while cap.isOpened():
            _, frame = cap.read()
            if resize:
                frame = cv2.resize(frame,self.size)

            classes, scores, boxes, _ = self.get_all_features(frame,nms,confidence) # get ALL detections

            if kind == 'detect':
                for (classid, score, box) in zip(classes, scores, boxes): # traverse each detection in the frame
                    if find and class_dict[classid[0]] not in find: # if the detected class is the one we want
                        continue
                    else:
                        color = COLORS[classid[0]] # nothing special. Just a color coding scheme for different classes classid is an array of 1 element
                        label = "%s : %f" % (class_dict[classid[0]], score)
                        cv2.rectangle(frame, box, color, 1)
                        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            elif kind == 'track':
                assert isinstance(find,str), "Input a single object type for Tracking, such as find ='person' etc"
                rectangles = []
                count = 0

                for (classid, score, box) in zip(classes, scores, boxes):
                    if (class_dict[classid[0]] == find) and (score > confidence):
                        color = COLORS[classid[0]] # nothing special. Just a color coding scheme for different classes classid is an array of 1 element
                        label = "%s : %f" % (class_dict[classid[0]], score)
                        cv2.rectangle(frame, box, color, 1)
                        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

                        x, y , w, h, = box[0], box[1], box[2], box[3]
                        box = (x,y,x+w,y+h)
                        rectangles.append(box) # Because rectangle object Tracker() looks for is x,y,x_end, y_end
                        count+=1

                if count_at_t: # show how many of those objects are present in the frame now
                    label = f"There are {count} {find}s in the frame right now"
                    cv2.putText(frame,label,(5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,0,0), 1)

                objects_dict = self.centroid_tracker.refresh(rectangles)
                
                if unique: # prints centroid of each frame
                    for (object_id, centroid) in objects_dict.items(): # get every object per frame
                        cv2.putText(frame, f"ID: {object_id}", (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        cv2.circle(frame, (centroid[0],centroid[1]), 1, (0, 0, 255), 2) # plot a dot in the middle of bounding box
   
            cv2.imshow("detections", frame)
            key = cv2.waitKey(1)
            if key==27 or key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


class HaarCascadeTracker():
    '''
    Class to detect objects using Haar Cascade. Using it with Tracking by merging with Tracker() class in above sample
    1 Haarfile detects only 1 type of features. You can try it with multiples too but you have to run a loop for all and then get detections
    '''
    def __init__(self,cascade:str,label:str):
        '''
        args:
            cascade_file = path to HaarCascade File. USe specific file based on the object to be detected
            label: Label to show on Boutding box. Face, Car etc
        '''
        self.cascade = cascade
        self.model = cv2.CascadeClassifier(self.cascade)
        self.label = label
        self.centroid_tracker = TrackCentroid()
    
    
    def get_all_feat(self,frame:np.ndarray,factor:float=1.5,neigh:int=5,smooth:bool=False):
        '''
        args:
            factor: Scale factor to use
            neigh: Minimum neighbours to use
            Smooth: Whether to blur the image for smoothning. Will use Gaussian Blur. Change it to Mean, Median or Bilateral filters or soem other
        out:
            list of ALL the bounding boxes. List of tuples of [(x,y,w,h),(x1,w1,h1,w1)]
        '''
        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if smooth:
            grey = cv2.GaussianBlur(grey,ksize=(3,3),sigmaX=0) # change these params for your image specific
        return self.model.detectMultiScale(grey,scaleFactor=factor, minNeighbors=neigh)   # return ALL the detected Features   

    
    def track(self,filepath:[bool,str]=None,crop:[bool,tuple,list]=(416,416)):
        '''
        args:
            filepath: file to be used the detection on. uses Camera if not given any filepath
            crop: Whether to crop or not
        '''
        cap = cv2.VideoCapture(0) if not filepath else cv2.VideoCapture(filepath)

        while cap.isOpened():
            _, frame = cap.read()
            if crop:
                frame = cv2.resize(frame,crop)
            
            features = self.get_all_feat(frame) # get all features

            rectangles = []
            count = 0
            for feat in features:
                x, y, w, h = feat[0], feat[1], feat[2], feat[3] # start x, start y, width, height of BB
                box = (x,y,x+w,y+h)
                rectangles.append(box)
                cv2.rectangle(frame, (x,y,w,h), (0,255,0), 1)
                cv2.putText(frame, self.label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
                count+=1

            objects_dict = self.centroid_tracker.refresh(rectangles)

            label = f"There are {count} {self.label}s in the frame right now"
            cv2.putText(frame,label,(5, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,0,0), 1)
            
            for (object_id, centroid) in objects_dict.items(): # get every object per frame
                cv2.putText(frame, f"ID: {object_id}", (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.circle(frame, (centroid[0],centroid[1]), 1, (0, 0, 255), 2) # plot a dot in the middle of bounding box

            cv2.imshow('Video',frame)

            key = cv2.waitKey(1)
            if key==27 or key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


class LineCrossingCounting():
    '''
    Class to count the objects which has crossed a certain line. Needs Bounding Box. Bounding Boxes can be from HaarCascade, DL Model or so
    '''
    ...


