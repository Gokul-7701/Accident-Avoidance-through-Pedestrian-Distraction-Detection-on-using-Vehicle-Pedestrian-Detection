import backbone
import Detection


#Place the input file address next line
input_video = "D:\Programs\Microsoft\Build Tools\projects\ML\GD/017.mp4" 

detection_graph, category_index = backbone.set_model('mask_rcnn_inception_v2_coco_2018_01_28', 'mscoco_label_map.pbtxt')

is_color_recognition_enabled = False # set it to true for enabling the color prediction for the detected objects
roi = 237 # roi line position
deviation = 4.5 # the constant that represents the object counting area
custom_object_name_1 = 'Vehicle' # set it to your custom object name
custom_object_name_2 = 'Pedestrian' 
targeted_objects_name = "car, pedestrian"

Detection.object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled)
