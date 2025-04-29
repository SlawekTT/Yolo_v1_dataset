My first attempt at recreating YOLO v1 algo - starting with custom dataset. 
It takes classic YOLO dataset (i.e. /train, /test, /valid with /images and /labels subdirs, each label text file follows the [c, xc, yc, w, h] schema).
Custom class transforms it into the form acceptable by yolo v1 - i.e. a grid of SxS imposed on the 448x448 image, xc, and yc now relative to grid cell (they are located in) UL corner, 
c = 1 (i.e. we are sure, there is an object), while a OneHotEncoded vecor is attached after [1, xc, yc, w, h]. 

Generaly, as an output we get SxSxL tensor, where L=5+num_classes. Here one may clearly observe (from the size of target tensor, that one grid cell can only detect one class - 
here also only one object per grid cell). If there is an object in the gird cell, there is a vector ([1, xc, yc, w, h, ohe_vec]) at the corresponding coordinates in the output. For no detections, 
there is just a vector of length L filled with zeros

More info on my Medium.com channel https://medium.com/@telega.slawomir.ai/yolo-v1-implementation-dataset-class-creation-0822eea4b7fa
