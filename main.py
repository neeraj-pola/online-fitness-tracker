import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a) # Start (hip)
    b = np.array(b) # Middle (knee)
    c = np.array(c) # End(Ankle)
    
    # 1 -> y coordinate
    # 0 -> x coordinate
    #y = tan(x) then x = arctan(y)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)

    #making a 360 degree rotation is not possible, so to set a range of 180 
    if angle > 180.0:
        angle = 360 - angle    
    return angle 

cap = cv2.VideoCapture("Data/KneeBendVideo.mp4")


# Video Characteristics
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)
fps = int(cap.get(cv2.CAP_PROP_FPS))


# Counter and State Variables
relax_counter = 0 
bent_counter = 0
counter = 0
stage = None
feedback = None
img_arr = []



if cap.isOpened() == False:
    print("Error in opening video stream or file")

#setup mediapipe instance
with mp_pose.Pose(min_detection_confidence = 0.6, min_tracking_confidence=0.6) as pose:
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            # cv2.imshow('Frame',frame)


            #recoloring the image to rgb for display
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            #no writing in this phase
            image.flags.writeable = False   

            #for detection in mediapipe
            results = pose.process(image)


            #allow access again to write and mark on the frames 
            image.flags.writeable = True
            #recolor back to bgr (keeping the perspective to process the image in cv2)
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)


            try:
                #render detections
                landmarks = results.pose_landmarks.landmark
                #making as an instance to avoid cumbersome
                obj = mp_pose.PoseLandmark
                # Get coordinates of interested landmarks (23->left hip, 25->left knee,27->left ankle,11->left shoulder)
                hip = [landmarks[obj.LEFT_HIP.value].x, landmarks[obj.LEFT_HIP.value].y]
                knee = [landmarks[obj.LEFT_KNEE.value].x, landmarks[obj.LEFT_KNEE.value].y]
                ankle = [landmarks[obj.LEFT_ANKLE.value].x, landmarks[obj.LEFT_ANKLE.value].y]
                shoulder = [landmarks[obj.LEFT_SHOULDER.value].x, landmarks[obj.LEFT_SHOULDER.value].y]

                angle = calculate_angle(hip,knee,ankle)

                # get coordinates of parts to draw on image
                a0 = int(ankle[0] * width)
                a1 = int(ankle[1] * height)

                k0 = int(knee[0] * width)
                k1 = int(knee[1] * height)

                h0 = int(hip[0] * width)
                h1 = int(hip[1] * height)

                cv2.line(image, (h0, h1), (k0, k1), (150, 255, 0), 2)
                cv2.line(image, (k0, k1), (a0, a1), (50, 255, 50), 2)
                cv2.circle(image, (h0, h1), 5, (0, 0, 0), cv2.FILLED)
                cv2.circle(image, (k0, k1), 5, (0, 0, 0), cv2.FILLED)
                cv2.circle(image, (a0, a1), 5, (0, 0, 0), cv2.FILLED)  

                cv2.putText(image, str(round(angle,4)), tuple(np.multiply(shoulder, [650, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                

                relax_time = (1 / fps) * relax_counter
                bent_time = (1 / fps) * bent_counter

                #Counter Logic
                if angle > 140:
                    relax_counter += 1
                    bent_counter = 0
                    stage = "Relaxed"
                    feedback = ""
                
                if angle < 140:
                    relax_counter = 0
                    bent_counter += 1
                    stage = "Bent"
                    feedback = ""
                
                # Sucessful rep
                if bent_time == 8:
                    counter += 1
                    feedback = 'Rep completed'
                    
                elif bent_time < 8 and stage == 'Bent':
                    feedback = 'Keep Your Knee Bent'
                
                else:
                    feedback = ""

            except:
                pass


            # Setup status box
            cv2.rectangle(image, (0,0), (int(width), 60), (150,255,0), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (12,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, str(counter), 
                        (10,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (105,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (105,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Feedback
            cv2.putText(image, 'FEEDBACK', (310,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, feedback, 
                        (315,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Bent Time
            cv2.putText(image, 'BENT TIME', (725,15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            
            cv2.putText(image, str(round(bent_time,2)), 
                        (725,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)  
                
            img_arr.append(image)

            cv2.imshow("solution",image)

            # Press esc to exit
            if cv2.waitKey(20) & 0xFF == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows() 

# Generate output video
out = cv2.VideoWriter('Output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(img_arr)):
    out.write(img_arr[i])
out.release()