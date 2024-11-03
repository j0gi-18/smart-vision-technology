import os
import cv2
import easyocr
from ultralytics import YOLO
from ultralytics import solutions
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("best.pt")
names = model.names
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

region_points = [(500, 850), (1500, 850), (1500, 200), (500, 200)]

# Video writer
video_writer = cv2.VideoWriter("object_cropping_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    show=True,
    region=region_points,
    model="best.pt",
)

idx = 0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(im0, line_width=2, example=names)
    
    # counting objects
    im0 = counter.count(im0)
    video_writer.write(im0)

    if boxes is not None:
        for box, cls in zip(boxes, clss):
            idx += 1
            annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

            crop_obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]

            # ocr implementation
            results = reader.readtext(crop_obj)

            text_line = " ".join([result[1] for result in results])
            print("detected text: " + text_line)


    cv2.imshow("ultralytics", im0)
    video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
