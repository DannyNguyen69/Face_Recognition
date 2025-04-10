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
  

yolo_face_dataset/  
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

# Main  
In the `main/` directory, there's a script named `train.py`. This file is responsible for customizing how the model is trained using the dataset you created.  
You can modify it to adjust training parameters such as the number of epochs, batch size, learning rate, or even change the model architecture.  


Once training is complete, navigate to the `best.pt` file located in the most recent training directory (e.g., `runs/detect/trainX/best.pt`, where `X` is the latest run number).
Then, use that path to fill in the model location as instructed in the `main.py` file.
