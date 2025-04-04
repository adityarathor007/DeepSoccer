import supervision as sv
from tqdm.notebook import tqdm
import numpy as np
import itertools
import torch
import cv2
from ultralytics import YOLO
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


start_time = time.time()

ball_id=0


src_video_path='data/tests/test_1/test_1_src.mp4'
    
detection_model_path = "runs/train/weights/best.pt"
detection_model = YOLO(detection_model_path).to(DEVICE)


only_balls_model_path = "runs/train_ball/weights/best.pt"
only_balls_model = YOLO(only_balls_model_path).to(DEVICE)


video_info=sv.VideoInfo.from_video_path(src_video_path)
# video_sink=sv.VideoSink(target_tracker_video_path,video_info=video_info)

BALL_ID = 0

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=25,
    height=21,
    outline_thickness=1
)

frame_generator = sv.get_video_frames_generator(src_video_path)

no_ball_init=0
no_ball_final=0


for i, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):

    result = detection_model(frame, conf=0.3)[0]
    detections = sv.Detections.from_ultralytics(result)

    ball_detections = detections[detections.class_id == BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    if len(ball_detections)==0:
            no_ball_init+=1
            print("No ball detected.")
            only_balls_result = only_balls_model(frame, conf=0.3)[0]
            only_balls_detections = sv.Detections.from_ultralytics(only_balls_result)
            ball_detections = only_balls_detections[only_balls_detections.class_id == ball_id]
    if len(ball_detections)==0:
        print("still no ball detected!")
        no_ball_final+=1
    else:
        max_conf_idx = ball_detections.confidence.argmax()
        print("Max confidence ball found in new model:",ball_detections.confidence.max())
        ball_detections = ball_detections[max_conf_idx:max_conf_idx+1]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    all_detections = detections[detections.class_id != BALL_ID]
    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    all_detections.class_id -= 1

    annotated_frame = frame.copy()
    annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=all_detections)
    annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)



    cv2.imshow("Real-Time Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


print(f"Ball not detected in {no_ball_init} frames in old model")
print(f"Ball not detected in {no_ball_final} frames in new model")
print(f"Balls detected in extra {no_ball_init-no_ball_final} frames in new model")