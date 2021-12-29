# Real-Time-Object-Detection

Object detetection plays a vital role in areas of computer vision ,which uses various machine learning techniques and deep learning algorithms. And it is a technique we use to identify the location of objects in an image. If there is a single object in the image and we want to detect that object, it is known as image localization.

We use OpenCV library which is used and utilized in the fields of robotics,computer Vision,etc.OpenCV has a bunch of pre-trained classifiers that can be used to identify objects such as trees, number plates, faces, eyes, etc. We can use any of these classifiers to detect the object as per our need.

1)I have used Common Objects in Context (COCO),which  is a database that aims to enable future research for object detection, instance segmentation, image captioning, and person keypoints localization.

           a)Coco Dataset9contains 91 common object names )
           b)ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt which is a  Config File for Trained MobileNet SSD Model(many versions are also available).
           c)Weights - frozen_inference_graph.pb -  derived from tensor flow
           
           
 2) In this project I have only used OpenCv and numpy libraries. Professinals and experts use TensorFlow and YOLO(You Look Only Once) for attaining more accuray and approxmimation. Whereas, this project is basically for beginners for better understanding ,who doesnt have preferable knowledge on Tensorflow and YOLO.    
