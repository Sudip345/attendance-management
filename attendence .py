""" a small projetct for attendance management using 
    1. open cv for object detction 
    2. haarcascade for face detction
    3. deepface for face recognization
    simpel and easy to understand , may add more features in future"""

import cv2 as cv 
from deepface import DeepFace as df
import numpy as np
import os
import time
import pandas as pd
from datetime import datetime


def mark_enter(roll, time, date): # mark a student as attended
    new_date = pd.DataFrame({
        "Roll": [roll],
        "Enter time": [time],
        "Exit time": [""],  # Empty string initially
        "date": [date]
    })
    new_date.to_csv("attendance.csv", mode="a", header=False, index=False)
    print(f"✅ Roll {roll} marked as attended at {time}.")
    return True


def mark_exit(roll, time, date):  # mark as left
    try:
        data = pd.read_csv("attendance.csv", dtype=str)

        index = -1
        for i, row in data.iterrows():
            if row['Roll'] == roll and row['date'] == date:
                index = i
                break

        if index == -1:
            print(f"⚠️ Roll {roll} on date {date} not found.")
            return False

        # Update "Exit time" column
        data.loc[index, "Exit time"] = time  

        # Save updated file
        data.to_csv("attendance.csv", index=False)

        print(f"✅ Roll {roll} marked as left at {time}.")
        return True
    
    except Exception as e:
        print(f"❌ Error {e}")
        return False

def check_if_marked(data, roll, date):  
    #Check if a student is already marked for attendance
    try:
        for _, row in data.iterrows():
            if row['date'] == date and row["Roll"] == roll:
                enter_time = row["Enter time"] != "" and not pd.isna(row["Enter time"])
                exit_time = row["Exit time"] != "" and not pd.isna(row["Exit time"])
                return enter_time, exit_time
        return False, False
    except Exception as e:
        print(f"❌ Error in check_if_marked: {e}")
        return False, False

        

def attendance_marker(roll):

    date = datetime.now().strftime("%d-%m-%Y")
    time = datetime.now().strftime("%H:%M")
    
    try:
        data = pd.read_csv("attendance.csv", dtype=str)
    except FileNotFoundError:
        print("⚠️ Attendance file not found. Creating a new one.")
        data = pd.DataFrame(columns=["Roll", "Enter time", "Exit time", "date"])
        data.to_csv("attendance.csv", index=False)

    enter_marked, exit_marked = check_if_marked(data, roll, date)
    fixed_time = datetime.strptime("01:00", "%H:%M").time() # preferred time , after this student will not be marked as attended 
    current_time = datetime.now().time()

    if enter_marked:
        if current_time>fixed_time:
            if exit_marked:
                print(f"⚠️ Roll {roll} is already marked as left for {date}.")
            else:
                mark_exit(roll, time, date)
        else :
            print(f"⚠️ Roll {roll} is already marked as attended for {date}.")
    else:
        if current_time>fixed_time:
            print("⚠️ Sorry u are too late, talk to your teachers for attendence")
        else:
            mark_enter(roll, time, date)




def get_roll(path):  # Extract the roll number assuming the naming format would be \....\roll_no
    path = path[0]
    i = len(path) - 1
    while i > -1 and path[i] not in ('/', '\\'):
        i -= 1
    roll = path[i+1:]
    roll = os.path.splitext(roll)[0]
    return roll

def match(img, db_path):
    exclude = set()
    count = 0
    roll = ""

    while count < 5:
        count += 1
        try:
            reslt = df.find(img, db_path, model_name="VGG-Face")
            reslt = reslt[0] if isinstance(reslt, list) else reslt
            reslt = reslt[~reslt["identity"].isin(exclude)]

            for i in range(len(reslt)):
                locations = reslt["identity"].to_numpy()
                roll = get_roll(locations)
                choice = input(f"Roll {roll} Press y to confirm else any Key: ")
                if choice.lower() == 'y':
                    return roll
                else:
                    roll = ""
                    exclude.add(str(locations[i]))

        except ValueError:
            print("Error while matching")
    
    return roll

def change_resolution(cap, size=0.55):
    height = int(2000 * size)
    width = int(1000 * size)
    cap.set(3, width)
    cap.set(4, height)
    return cap

def get_photo(cap):
    haar_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    flag=0
    while True:
      isTrue, frame = cap.read()
      frame= cv.flip(frame,1)
      gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      det_face = haar_cascade.detectMultiScale(gray, 1.11, 5)
      for (x, y, w, h) in det_face:
         cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
         face = frame[y:y+h, x:x+w]
         cv.imwrite("live.jpg", face)
        #  print("Face Captured and Saved as 'live.jpg'")
         flag =1
      cv.imshow("Face detection",frame)
      if cv.waitKey(20) == ord('d'):
          if flag == 1:
              return True
          else :
              return False
      

if __name__ =="__main__":
    try:
        cap = cv.VideoCapture(0)
        cap = change_resolution(cap)

        if os.path.exists("live.jpg"):
            os.remove("live.jpg")

        print("Press 'd' after few seconds of successful face detection")
        while True:
            if get_photo(cap):
                roll = match("live.jpg", r"C:\Users\sudip\Desktop\new projects\phostos\known")
                if roll == "":
                    print("Match not found, try again or get help from faculties.")
                else:
                    attendance_marker(roll)
    
            if cv.waitKey(1000) == ord('q'):
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("exiting...")
    cap.release()
    cv.destroyAllWindows()
