import cv2

# path of video file
video_path = "/home/nivetheni/front_cam.mp4"

# Open video file
video = cv2.VideoCapture(video_path)

# number of frames in video
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Convert frame to image and save to file
for i in range(frame_count):
    ret, frame = video.read()
    if ret:
        image_path = f"in1/image_{i}.jpg"
        cv2.imwrite(image_path, frame)

# Close video file
video.release()