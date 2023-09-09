import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime

# Load a pre-trained face recognition model
def load_face_recognition_model():
    known_face_encodings = []
    known_face_names = []

    # Load known faces and their names from your array of objects
    # Each object should contain an image and a name
    objects = [
        {"name": "vishesh", "image_path": "vishesh.jpg"},
        {"name": "shub", "image_path": "shub.jpg"},
        {"name": "aryan", "image_path": "aryan.jpg"},
        {"name": "himan", "image_path": "himanshu.jpg"},
        {"name": "sir", "image_path": "sir.jpg"},
        # Add more objects with names and image paths
    ]

    for obj in objects:
        image = face_recognition.load_image_file(obj["image_path"])
        face_encoding = face_recognition.face_encodings(image)[0]  # Assuming one face per image
        known_face_encodings.append(face_encoding)
        known_face_names.append(obj["name"])

    return known_face_encodings, known_face_names

def main():
    # Load the pre-trained face recognition model
    known_face_encodings, known_face_names = load_face_recognition_model()

    # Initialize OpenCV's video capture
    cap = cv2.VideoCapture(0)

    # Create a unique Excel sheet filename using the current date and time
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    excel_filename = f"attendance_{timestamp}.xlsx"

    # Create an empty DataFrame to store attendance
    attendance_df = pd.DataFrame(columns=["Name", "Time"])

    # Create a dictionary to keep track of recorded names
    recorded_names = {}

    while True:
        ret, frame = cap.read()

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for face_encoding in face_encodings:
            # Compare the current face encoding to known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # If a match is found and the name hasn't been recorded yet, record attendance
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                if name not in recorded_names:
                    # Record attendance
                    now = datetime.now()
                    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    attendance_df = pd.concat([attendance_df, pd.DataFrame({"Name": [name], "Time": [current_time]})], ignore_index=True)
                    recorded_names[name] = True

            # Draw a rectangle and label the face on the frame
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save attendance to the unique Excel file
    attendance_df.to_excel(excel_filename, index=False)

    # Release the video capture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
