# Face_Recognition
This is a project use cascade_classifier in opencv to make a dataset for training YOLO. 

# Requestments
You need to have these libaries:  
      tkinter :  pip install tkinter  
			yaml : pip install yaml  
	    opencv : pip install opencv-python  
		  ultralytics : pip install ultralytics ( to use YOLOv8n or similar model )  
		  random  
		  cv zone : pip install cvzone  
		  math

# Labeling_app
"I use a Cascade Classifier to detect human faces, which then appear as bounding boxes around the faces in OpenCV. After that, the program saves the images and their corresponding labels as a YOLO dataset, along with a .yml file and a .txt file containing the number of classes.  
## Dataset structure:

Dưới đây là cấu trúc thư mục của dataset:  

dataset/  
├── images/  
│   ├── train/  
│   │   ├── img1.jpg  
│   │   ├── img2.jpg  
│   │   └── ...  
│   └── valid/  
│       ├── img101.jpg  
│       ├── img102.jpg  
│       └── ...  
├── labels/  
│   ├── train/  
│   │   ├── img1.txt  
│   │   ├── img2.txt  
│   │   └── ...  
│   └── valid/  
│       ├── img101.txt  
│       ├── img102.txt  
│       └── ...  
├── data.yaml  
└── classes.txt  
