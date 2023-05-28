import cv2
import hashlib

#A basic search through all the frames in a video using hashtables(because of O(1) time complexity)
#if a frame with already read hash values is found then we can see it as dupilicate

# Open the video file
cap = cv2.VideoCapture('Data/KneeBendVideo.mp4')

# Create an empty list to store the frames
frames = []

# Read the frames from the video, one by one
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Compute the hash value for the frame
    hash_value = hashlib.sha256(frame).hexdigest()
    
    # Add the frame and its hash value to the list
    frames.append((frame, hash_value))
    # print(hash_value)

# Loop through the list of frames and compare the hash values
for i in range(len(frames)):
    for j in range(i+1, len(frames)):
        if frames[i][1] == frames[j][1]:
            print("Found duplicate frames at index {} and {}".format(i, j))