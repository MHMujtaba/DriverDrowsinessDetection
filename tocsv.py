# referenced https://github.com/Mahima18/AI-Project-Final-from-Team-9/tree/master for getting an idea about implementaion.
import cv2
import dlib
import numpy as np
import pandas as pd
import listoffiles as lf
import listoffiles5 as lf5
import listoffiles10 as lf10
# Load face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Path to the pre-trained facial landmark detector
part_id =1
class_lab = 0
def eye_aspect_ratio(eye):
    
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    
    C = np.linalg.norm(eye[0] - eye[3])
    # Eye Aspect Ratio
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    
    C = np.linalg.norm(mouth[0] - mouth[6])
    # Mouth Aspect Ratio
    mar = (A + B) / (2.0 * C)
    return mar

def calculate_face_angle(shape):
    
    nose = shape.part(34)
    
    u = np.array([shape.part(49).y-nose.y,shape.part(49).x-nose.x])
    v = np.array([shape.part(55).y-nose.y,shape.part(55).x-nose.x])
    
    theta = np.dot(u,v) / (np.linalg.norm(u)* np.linalg.norm(v))
    pitch_angle_deg = np.arccos(theta)
    return np.degrees(pitch_angle_deg)

def getframes(frame_index):
    cap.set(cv2.CAP_PROP_POS_MSEC, 180000 + frame_index*1000)
    numframes = 0
    
    framesAvail,currFrame = cap.read()
    frames = []
    frames.append(currFrame)

    while (numframes <9) and (framesAvail is True):
        framesAvail,currFrame = cap.read()
        frames.append(currFrame)
        numframes+=1

    return framesAvail,frames

# skip_files=[3,6,16,18,31,37,44,46,48,49,58,60]
# For loop to iterate over each video and extarct the facial features using the functions above.
# Outputs a CSV file for each video.
for id in range(0,len(lf.list_files)):
    
    part_id_list = []
    ear_list = []
    mar_list = []
    mouth_eye_list = []
    angle_list = []
    label_list = []
    i  = 0
    skip = 0
    frame_index = 1
    part_id = id + 1
#    Start reading video at 3 min mark
    cap = cv2.VideoCapture(lf.list_files[id])
    cap.set(cv2.CAP_PROP_POS_MSEC, 180000)
    print("capturing for vid",part_id)
    while cap.isOpened():
        
        frames_left, frame = cap.read()
        if not frames_left:
            print("breaking...")
            break
            
        else:    
            # Get 10 frames from 1 second.
            avail,frames_list = getframes(frame_index)
            if not avail:
                break
            frame_index += 1
            for frame in frames_list:
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                # Get facial landamarks and feature values.
                for face in faces:
                    landmarks = predictor(gray, face)
                    angle = calculate_face_angle(landmarks)
                    landmarks = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
                    
                    
                    left_eye = landmarks[36:42]
                    right_eye = landmarks[42:48]
                    mouth = landmarks[48:68]

                    ear = eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)
                    mar = mouth_aspect_ratio(mouth)
                    mouth_eye = mar/ear
                    # Append values to list of values.
                    part_id_list.append(part_id)
                    ear_list.append(ear)
                    mar_list.append(mar)
                    mouth_eye_list.append(mouth_eye)
                    angle_list.append(angle)
                    label_list.append(class_lab)
                    
                if(i%100==0)  : 
                    print("frame",i)
                    
                i+=1
        if i == 4200:
            print("breaking at 4200 frames....")
            break
        

    # Save EAR, MAR, MOE and angle to CSV
    # print(data)
    data = {"Part_ID":part_id_list,"EAR": ear_list, "MAR": mar_list,"MOE":mouth_eye_list,"Angle":angle_list,"Label":label_list}
    df = pd.DataFrame(data)
    df.to_csv(f"datasets/EMYA_{part_id}_{class_lab}.csv", index=False)

    cap.release()
    cv2.destroyAllWindows()
# Repeate steps for class 5 and 10.
part_id =1
class_lab=5    
for id in range(0,len(lf5.list_files5)):
    
    part_id_list = []
    ear_list = []
    mar_list = []
    mouth_eye_list = []
    angle_list = []
    label_list = []
    i  = 0
    skip = 0
    frame_index = 1
    part_id = id + 1
    # if skip_files.__contains__(part_id):
    #     continue
    # print("vidoe->",lf.list_files[id])
    cap = cv2.VideoCapture(lf5.list_files5[id])
    cap.set(cv2.CAP_PROP_POS_MSEC, 180000)
    print("capturing for vid",part_id)
    while cap.isOpened():
        
        frames_left, frame = cap.read()
        if not frames_left:
            print("breaking...")
            break
            
        else:    
            avail,frames_list = getframes(frame_index)
            if not avail:
                break
            frame_index += 1
            for frame in frames_list:
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                
                for face in faces:
                    landmarks = predictor(gray, face)
                    angle = calculate_face_angle(landmarks)
                    landmarks = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
                    # print(landmarks[0])
                    
                    left_eye = landmarks[36:42]
                    right_eye = landmarks[42:48]
                    mouth = landmarks[48:68]

                    ear = eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)
                    mar = mouth_aspect_ratio(mouth)
                    mouth_eye = mar/ear
                    part_id_list.append(part_id)
                    ear_list.append(ear)
                    mar_list.append(mar)
                    mouth_eye_list.append(mouth_eye)
                    angle_list.append(angle)
                    label_list.append(class_lab)
                    # print(ear," " ,mar)
                # print(cap.isOpened())
                if(i%100==0)  : 
                    print("frame",i)
                    
                i+=1
        if i == 4200:
            print("breaking at 4200 frames....")
            break
        

    # Save EAR and MAR to CSV
    # print(data)
    data = {"Part_ID":part_id_list,"EAR": ear_list, "MAR": mar_list,"MOE":mouth_eye_list,"Angle":angle_list,"Label":label_list}
    df = pd.DataFrame(data)
    df.to_csv(f"datasets/EMYA_{part_id}_{class_lab}.csv", index=False)

    cap.release()
    cv2.destroyAllWindows()
        

part_id =1
class_lab=10    
for id in range(0,len(lf10.list_files10)):
    
    part_id_list = []
    ear_list = []
    mar_list = []
    mouth_eye_list = []
    angle_list = []
    label_list = []
    i  = 0
    skip = 0
    frame_index = 1
    part_id = id + 1
    # if skip_files.__contains__(part_id):
    #     continue
    # print("vidoe->",lf.list_files[id])
    cap = cv2.VideoCapture(lf10.list_files10[id])
    cap.set(cv2.CAP_PROP_POS_MSEC, 180000)
    print("capturing for vid",part_id)
    while cap.isOpened():
        
        frames_left, frame = cap.read()
        if not frames_left:
            print("breaking...")
            break
            
        else:    
            avail,frames_list = getframes(frame_index)
            if not avail:
                break
            frame_index += 1
            for frame in frames_list:
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                
                for face in faces:
                    landmarks = predictor(gray, face)
                    angle = calculate_face_angle(landmarks)
                    landmarks = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
                    # print(landmarks[0])
                    
                    left_eye = landmarks[36:42]
                    right_eye = landmarks[42:48]
                    mouth = landmarks[48:68]

                    ear = eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)
                    mar = mouth_aspect_ratio(mouth)
                    mouth_eye = mar/ear
                    part_id_list.append(part_id)
                    ear_list.append(ear)
                    mar_list.append(mar)
                    mouth_eye_list.append(mouth_eye)
                    angle_list.append(angle)
                    label_list.append(class_lab)
                    # print(ear," " ,mar)
                # print(cap.isOpened())
                if(i%100==0)  : 
                    print("frame",i)
                    
                i+=1
        if i == 4200:
            print("breaking at 4200 frames....")
            break
        

    # Save EAR and MAR to CSV
    # print(data)
    data = {"Part_ID":part_id_list,"EAR": ear_list, "MAR": mar_list,"MOE":mouth_eye_list,"Angle":angle_list,"Label":label_list}
    df = pd.DataFrame(data)
    df.to_csv(f"datasets/EMYA_{part_id}_{class_lab}.csv", index=False)

    cap.release()
    cv2.destroyAllWindows()
        



