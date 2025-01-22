import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys


threshold = 0.5
model_height,model_width = 320,320

def find_best_boxes(boxes): # 300,1200,4800
    bounding_boxes = []
    pro_value_class = []
    class_name_index = []

    for i in boxes:
        for j in i:
            prob_values = j[5:] # all 80 class probability values
            class_index = np.argmax(prob_values) # index of high class value
            confidence = prob_values[class_index] # finding the class value

            if confidence > threshold:
                w, h = int(j[2] * model_width), int(j[3] * model_height)
                # print(w , h)
                # finding x and y
                x, y = int(j[0] * model_width - w / 2), int(j[1] * model_height - h / 2)

                bounding_boxes.append([x,y,w,h])
                class_name_index.append(class_index)
                pro_value_class.append(confidence)

    final_boxes = cv2.dnn.NMSBoxes(bounding_boxes,pro_value_class,threshold,.6)
    return final_boxes,bounding_boxes,pro_value_class,class_name_index


def final_detection(main_boxes, total_boxes, each_box_acc, each_box_class_index, height_ratio, width_ratio):
    detected_classes = {}  # Dictionary to store the count of each class

    for k in main_boxes:
        x, y, w, h = total_boxes[k]
        x = int(x * width_ratio)
        y = int(y * height_ratio)
        w = int(w * width_ratio)
        h = int(h * height_ratio)
        conf = each_box_acc[k]
        class_name = coco_class_names[each_box_class_index[k]]

        # Count how many times each class is detected
        if class_name in detected_classes:
            detected_classes[class_name] += 1
        else:
            detected_classes[class_name] = 1

        text = str(class_name) + ':' + str("%.2f" % conf)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame, text, (x, y - 3), font, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y), (w + x, h + y), (0, 255, 0), 2)

    # Display the class count on the top-left corner
    class_count_text = 'Detected Classes: ' + str(len(detected_classes))
    cv2.putText(frame, class_count_text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
    y_set=50
    for class_name,count in  detected_classes.items():
        class_count_text=f'{class_name}:{count}'
        cv2.putText(frame,class_count_text,(20,y_set),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
        y_set+=30
cap = cv2.VideoCapture('yolo_test.mp4')


coco_class_names = []
with open('class_names','r') as f:
    for i in f.readlines():
        coco_class_names.append(i.strip())

v3_neural_network = cv2.dnn.readNetFromDarknet('yolov4.cfg','yolov4.weights') # loading architecture and weights
complete_layers = v3_neural_network.getLayerNames() # checking complete architecture
v3_neural_network.getUnconnectedOutLayersNames() # yolo outcome layers
index_yolo_layers = v3_neural_network.getUnconnectedOutLayers() # index of yolo layers
final_yolo_layers = [complete_layers[j-1] for j in index_yolo_layers] # fetching proper yolo layers

while True:
    res , frame = cap.read()
    if res == True:
        original_image_height, original_image_width = frame.shape[0], frame.shape[1]
        input_image = cv2.dnn.blobFromImage(frame,1/255,(320,320),True,crop=False) # (1,channel,height,width)
        v3_neural_network.setInput(input_image) # input image to yolov3
        nn_outcome = v3_neural_network.forward(final_yolo_layers) # outcomes from 3 yolo layers
        f_box,t_box,box_acc,index_class = find_best_boxes(nn_outcome) # calling the function
        final_detection(f_box,t_box,box_acc,index_class,original_image_height/320 , original_image_width/320)
        cv2.imshow('car_image',frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()