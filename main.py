import sys
import torch
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from sahi.predict import get_sliced_prediction,AutoDetectionModel,visualize_object_predictions
import cv2
from norfair import Detection, Tracker, Video, draw_boxes, draw_tracked_boxes
from norfair.filter import OptimizedKalmanFilterFactory
import pytube
from pytube import YouTube
import warnings
warnings.filterwarnings("ignore")

device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")

video_links= ["""https://www.youtube.com/watch?v=WeF4wpw7w9k""",
              """https://www.youtube.com/watch?v=2NFwY15tRtA""",
              """https://www.youtube.com/watch?v=5dRramZVu2Q""",
              """https://www.youtube.com/watch?v=2hQx48U1L-Y"""]

input_video_file_1 = f"{sys.path[0]}/Cyclist and vehicle Tracking - 1.mp4" # 1st video
input_video_file_2 = f"{sys.path[0]}/Cyclist and vehicle tracking - 2.mp4" # 2nd video
input_video_file_3 = f"{sys.path[0]}/Drone Tracking Video.mp4" # 3rd video
input_video_file_4 = f"{sys.path[0]}/Dji Mavic air 2 drone using litchi app with follow me mode on a bike occluded by trees.mp4" # 4th video

output_video_path = f"{sys.path[0]}"

model = "yolov8"
weights= f"{sys.path[0]}/mymodel_70_last.pt"

required_classes1= ["car","bicycle"]
required_classes2= ["car","pedestrian"]
required_classes3= ["pedestrian","person"]
required_classes4= ["pedestrian","person","bicycle"]

# Kalman Filter Parameters for 2nd Video

model_threshold_2 = 0.1
slice_height_2= 200 
slice_width_2= 400

# Kalman Filter Parameters for 3rd Video

model_threshold_3 = 0.3
slice_height_3= 450 
slice_width_3= 350

# Kalman Filter Parameters for 4th Video

model_threshold_4 = 0.1
slice_height_4= 170
slice_width_4= 120


def download_video(link):
    print(link)
    youtubeObject = pytube.YouTube(link)
    youtubeObject = youtubeObject.streams.get_lowest_resolution()
    try:
        youtubeObject.download()
    except:
        print("An error has occurred")
    print("Download is completed successfully")


def detect(file_path,model,weights,threshold,device,plot=False,only_model=False):

    detection_model= AutoDetectionModel.from_pretrained(model_type= model,
                                        model_path=weights,
                                        device=device,
                                        confidence_threshold=threshold)
    
    if only_model:
        return detection_model
    
    if file_path:
        result = get_sliced_prediction(
            file_path,
            detection_model,
            slice_width=400,
            slice_height=200,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3
        )

        if plot:
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            numpydata = np.asarray(img_converted)
            frame= visualize_object_predictions(
                numpydata, 
                object_prediction_list=result.object_prediction_list,
                hide_labels=1,
                output_dir=sys.path[0],
                file_name=f"test_img_{file_path.split('/')[-1].split('.')[0]}",
                export_format="png"
                )
            plot_img= cv2.imread(f"{sys.path[0]}/test_img_{file_path.split('/')[-1].split('.')[0]}.png",cv2.IMREAD_UNCHANGED)
            img_conv= cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_conv)
            plt.show()
        return result

def Apply_Kalman_Filter(input_video_file,output_video_path,model,model_weights,required_labels,image_file_path=None,model_threshold=0.5,distance_threshold=30,hit_count_max=20,
                        slice_height=300,slice_width=300,overlap_ratio=0.3,distance_function="euclidean",skip_period=1,device=device):

    detection_model= detect(file_path=image_file_path,
                            model=model,
                            weights=model_weights,
                            threshold=model_threshold,
                            device= device,
                            only_model=True)

    tracker= Tracker(distance_function=distance_function,
                    hit_counter_max=hit_count_max,
                    filter_factory=OptimizedKalmanFilterFactory(),
                    distance_threshold=distance_threshold)

    tracking_video= Video(input_path=input_video_file,output_path=output_video_path)

    for i,frame in enumerate(tracking_video):
        if i%skip_period == 0:
            detections=[]
            result = get_sliced_prediction(frame,detection_model,
                                            slice_height=slice_height,
                                            slice_width=slice_width,
                                            overlap_height_ratio=overlap_ratio,
                                            overlap_width_ratio=overlap_ratio,
                                            verbose=0)

            det_objs= [i for i in result.object_prediction_list if i.category.name in required_labels and i.score.value >model_threshold]

            for det_s in det_objs:

                bbox_coords = det_s.bbox

                bbox_np = np.array([tuple(bbox_coords.to_voc_bbox()[0:2]),tuple(bbox_coords.to_voc_bbox()[2:])])

                detections.append(Detection(points= bbox_np,
                                scores=np.array([det_s.score.value for _ in bbox_np]),
                                label=det_s.category.name))

                tracked_objects= tracker.update(detections=detections,
                                                period=skip_period)
                draw_boxes(frame,detections,draw_labels=True,draw_scores=True,line_color="white",text_color="white",text_size=0.2)
                draw_tracked_boxes(frame,tracked_objects,draw_labels=False,draw_box=True)
        else:
            tracked_objects=tracker.update()
        
        tracking_video.write(frame)

def train(model_name,data,epochs,image_size):
    model = YOLO(model_name)

    results = model.train(
        data=data,
        epochs = epochs,
        imgsz=image_size)
    
    return results

def main():

    #train_model= train(model_name,data,epochs,image_size)
    n=0
    for link in video_links:
        if n==1:
            break
        n+=1
        download_video(link)

    Apply_Kalman_Filter(input_video_file=input_video_file_1,output_video_path=output_video_path,model=model,
                    model_weights=weights,required_labels=required_classes1,model_threshold=0.25)
    
    Apply_Kalman_Filter(input_video_file=input_video_file_2,output_video_path=output_video_path,model=model,model_weights=weights,
                        required_labels=required_classes2,model_threshold=model_threshold_2,slice_height=slice_height_2,slice_width=slice_width_2)
    
    Apply_Kalman_Filter(input_video_file=input_video_file_3,output_video_path=output_video_path,model=model,model_weights=weights,
                        required_labels=required_classes3,model_threshold=model_threshold_3,slice_height=slice_height_3,slice_width=slice_width_3)

    Apply_Kalman_Filter(input_video_file=input_video_file_4,output_video_path=output_video_path,model=model,model_weights=weights,
                        required_labels=required_classes4,model_threshold=model_threshold_4,slice_height=slice_height_4,slice_width=slice_width_4)

if __name__ == "__main__":
    main()