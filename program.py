import face_recognition
import cv2
import numpy as np  
import csv
import os
from datetime import datetime   

video_capture = cv2.VideoCapture(0)
rock_image = face_recognition.load_image_file("photo/rock.jpg")
rock_face_encoding = face_recognition.face_encodings(rock_image)[0]

ye_image = face_recognition.load_image_file("photo/ye.jpeg")
ye_face_encoding = face_recognition.face_encodings(ye_image)[0]

ksi_image = face_recognition.load_image_file("photo/ksi.jpeg")
ksi_face_encoding = face_recognition.face_encodings(ksi_image)[0]

known_face_encodings = [
    rock_face_encoding,
    ye_face_encoding,
    ksi_face_encoding
]
known_face_names = [
    "rock",
    "ye",
    "ksi"
]
students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")   

f = open(current_date+'.csv','w+',newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time]) 


    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
f.close()