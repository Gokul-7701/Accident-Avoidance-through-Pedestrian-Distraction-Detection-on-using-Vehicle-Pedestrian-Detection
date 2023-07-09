import tensorflow as tf
import math,cv2
import numpy as np
import visualize
from tensorflow import keras
from tensorflow.keras import regularizers
from keras.preprocessing import image
from tensorflow.keras.models import model_from_json


def object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled):

        # input video
        cap = cv2.VideoCapture(input_video)

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.mp4', fourcc, fps, (width, height))


        with detection_graph.as_default():
          with tf.compat.v1.Session(graph=detection_graph) as sess:
            #Distraction Model Insertion
            model = keras.Sequential([keras.layers.Conv2D(64,kernel_size=3,activation='relu',padding='same',kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15), bias_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15),input_shape=(48, 48, 3)),
                                    keras.layers.BatchNormalization(momentum=0.1, epsilon=0.1),
                                    keras.layers.AvgPool2D(pool_size=(2, 2), padding='same'),
                                    keras.layers.Flatten(),
                                    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15), bias_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15)),
                                    keras.layers.Dense(2, activation='softmax')])

            model.compile(optimizer="rmsprop",loss='categorical_crossentropy', metrics=['accuracy'])
            model.load_weights("D:\Programs\Microsoft\Build Tools\projects\ML\Distract03.h5")

            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            prev_box = np.zeros(shape=(100, 4))
            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame
                image_np_expanded = np.expand_dims(input_frame, axis=0)
                width = len(input_frame)
                height = len(input_frame[0])
                
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                x , y = boxes[0].shape
                
                classes_1 = []
                scores_1 = []
                row = []
                c = []

                for i in range(0,x):
                    ymin = boxes[0][i][0]
                    xmin = boxes[0][i][1]
                    ymax = boxes[0][i][2]
                    xmax = boxes[0][i][3]

                    row.append(0.4)
                    c.append(0)

                    if ymin==0.0 or xmin==0.0 or xmax==0.0 or ymax==0.0:
                        prev_box = boxes[0]
                        continue
                    if not (classes[0][i]==1.0 or classes[0][i]==3.0 or scores[0][i]!=0.0):
                        prev_box = boxes[0]
                        continue
                    
                    if scores[0][i]<0.55:
                        prev_box = boxes[0]
                        continue

                    box = [0.0,0.0,0.0,0.0]
                    sum = 10.0

                    #Selecting the region proposal related to the previous region proposals
                    for j in range(0,x):
                        prev_ymin = prev_box[j][0]
                        prev_xmin = prev_box[j][1]
                        prev_ymax = prev_box[j][2]
                        prev_xmax = prev_box[j][3]

                        temp_sum = abs(prev_ymin-ymin)+abs(prev_ymax-ymax)+abs(prev_xmax-xmax)+abs(prev_xmin-xmin)
                        if temp_sum < sum:
                            box = [prev_ymin, prev_xmin, prev_ymax, prev_xmax]
                            sum = temp_sum

                    if sum > 0.2:
                        prev_box = boxes[0]
                        continue

                    ymin = boxes[0][i][0]*width
                    xmin = boxes[0][i][1]*height
                    ymax = boxes[0][i][2]*width
                    xmax = boxes[0][i][3]*height

                    prev_ymin = box[0]*width
                    prev_xmin = box[1]*height
                    prev_ymax = box[2]*width
                    prev_xmax = box[3]*height
                    
                    
                    r = []
                    ya = int(ymin)
                    yb = int(ymax)
                    xa = int(xmin)
                    xb = int(xmax)

                    #Making Sure region proposal size is 48*48
                    if (yb-ya)>48:
                        diff = abs(yb-ya)
                        diff = abs(48-diff)
                        if(diff%2==1):
                            yb = yb-1
                        
                        diff = diff/2
                        ya = ya+int(diff)
                        yb = yb-int(diff)
                    elif(yb-ya)==48:
                        continue
                    else:
                        diff = abs(yb-ya)
                        diff = abs(48-diff)

                        while(diff>1 and ya>0 and yb<width-1):
                            ya = ya-1
                            yb = yb+1
                            diff = diff-2
                        
                        while(diff>0 and ya>1):
                            ya = ya-1
                            diff = diff-1
                        while(diff>0 and yb<width-1):
                            yb = yb+1
                            diff = diff-1

                    if (xb-xa)>48:
                        diff = abs(xb-xa)
                        diff = abs(48-diff)
                        if(diff%2==1):
                            xb = xb-1
                        diff = diff/2
                        xa = xa+int(diff)
                        xb = xb-int(diff)
                    elif(xb-xa)==48:
                        continue
                    else:
                        diff = abs(xb-xa)
                        diff = abs(48-diff)
                        while(diff>1 and xa>0 and xb<height-1):
                            xa = xa-1
                            xb = xb+1
                            diff = diff-2
                        
                        while(diff>0 and xa>1):
                            xa = xa-1
                            diff = diff-1
                        while(diff>0 and xb<height-1):
                            xb = xb+1
                            diff = diff-1
                        
                    #Selecting only pedestrian and predicting whether distracted or not.
                    #elif statement make sure car is considered for visualization
                    if classes[0][i]==1.0:
                        for j in range(ya,yb):
                            r.append(frame[j][xa:xb])

                        img = r
                        img = image.img_to_array(img)
                        img = np.expand_dims(img, axis=0)
                        img = np.vstack([img])
                        cl = model.predict(img)
                        
                        row[i]=cl[0][np.argmax(cl)]
                        c[i] = np.argmax(cl)+1

                    elif classes[0][i]==3.0:
                        row[i]=1
                        c[i]=3
                    
                    else:
                        row[i]=0
                        c[i]=0


                    #Path prediction begins
                    if xmin+xmax+ymax+ymin == 0.0 or prev_xmax+prev_xmin+prev_ymax+prev_ymin==0.0:
                        prev_box = boxes[0]
                        continue

                    if xmin == prev_xmin or xmax==prev_xmax or prev_xmin==0.0 or prev_xmax==0 or xmin==0.0 or xmax == 0.0:
                        prev_box = boxes[0]
                        continue

                    m_1 = (ymax-prev_ymax)/(xmin-prev_xmin)
                    m_2 = (ymax-prev_ymax)/(xmax-prev_xmax)

                    c_1 = ymax - (m_1*xmin)
                    c_2 = ymax - (m_2*xmax)

                    if math.isnan(m_1):
                        m_1 = 0.0
                    if math.isnan(m_2):
                        m_2 = 0.0

                    y_1_1 = c_1
                    sp_1 = (0,int(y_1_1))
                    if y_1_1 < 0:
                        x_1 = (-c_1)/m_1
                        sp_1 = (int(x_1),0)
                    elif y_1_1 >= width:
                        x_1 = (width-c_1)/m_1
                        if math.isnan(x_1):
                            x_1 = 0.0
                        sp_1 = (int(x_1),width)

                    y_1_2 = m_1*height + c_1
                    ep_1 = (height,int(y_1_2))
                    if y_1_2 < 0:
                        x_1 = (-c_1)/m_1
                        ep_1 = (int(x_1),0)
                    elif y_1_2 >= width:
                        x_1 = (width-c_1)/m_1
                        if math.isnan(x_1):
                            x_1 = 0.0
                        ep_1 = (int(x_1),width)


                    thickness = 2
                    if classes[0][i]==3.0:
                        cv2.line(input_frame, sp_1, ep_1, (0, 0, 255), thickness)
                    elif classes[0][i]==1.0:
                        cv2.line(input_frame, sp_1, ep_1, (0, 255, 0), thickness)

                    y_2_1 = c_2
                    sp_2 = (0,int(y_2_1))
                    if y_2_1 < 0:
                        x_2 = (-c_2)/m_2
                        sp_2 = (int(x_2),0)
                    elif y_2_1 >= width:
                        x_2 = (width-c_2)/m_2
                        if math.isnan(x_2):
                            x_2 = 0.0
                        sp_2 = (int(x_2),width)

                    y_2_2 = m_2*height + c_2
                    ep_2 = (height,int(y_2_2))
                    if y_2_2 < 0:
                        x_2 = (-c_2)/m_2
                        ep_2 = (int(x_2),0)
                    elif y_2_2 >= width:
                        x_2 = (width-c_2)/m_2
                        if math.isnan(x_2):
                            x_2 = 0.0
                        ep_2 = (int(x_2),width)

                    #Line drawn after calculation
                    thickness = 2
                    if classes[0][i]==3.0:
                        cv2.line(input_frame, sp_2, ep_2, (0, 0, 255), thickness)
                    elif classes[0][i]==1.0:
                        cv2.line(input_frame, sp_2, ep_2, (0, 255, 0), thickness)
                    
                    
                #Addition of distraction model output for visualization.
                classes_1.append(c)
                scores_1.append(row)
                category_index_1 = {
                    0:{'id':0, 'name':'NP & NV'},
                    1:{'id':1, 'name':'Distracted'},
                    2:{'id':2, 'name':'Not Distracted'},
                    3:{'id':3, 'name':'Vehicle'}
                }

                counter, csv_line, counting_result = visualize.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes_1).astype(np.int32),
                                                                                                      np.squeeze(scores_1),
                                                                                                      category_index_1,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
                
                
                output_movie.write(input_frame)
                # print ("writing frame")
                cv2.imshow('Pedestrian-Vehicle Collision Detection',input_frame)
                # time.sleep(1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                prev_box = boxes[0]

            cap.release()
            cv2.destroyAllWindows()