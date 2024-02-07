import cv2
from ultralytics import YOLO

# model import
model = YOLO('yolov8m-pose.pt')

# # video capture
# vidcap = cv2.VideoCapture(0)

# if not vidcap.isOpened():
#     print("Error: Couldn't open camera.")
#     exit()

# cv2.namedWindow("Live Stream", cv2.WINDOW_NORMAL)

# while vidcap.isOpened():
#     # read a frame from the camera
#     ret, frame = vidcap.read()

#     # break the loop if reading the frame fails
#     if not ret:
#         print("Error: Failed to read frame.")
#         break

#     ## MODIFY FRAME FROM THIS POINT ON
#     result = model(frame)
    

#     # display the frame in the window
#     cv2.imshow("Live Stream", frame)

#     # break the loop if 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # release the camera and close the window
# vidcap.release()
# cv2.destroyAllWindows()

result = model(source = 0, show = True, conf = 0.5, save = False)
