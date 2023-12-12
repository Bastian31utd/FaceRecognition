import cv2
from deepface import DeepFace
import time

videoCapture = cv2.VideoCapture(0)
if not videoCapture.isOpened():
    print("Không thể kết nối với camera")
    exit()

checkPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(checkPath)

last_recognition_time = time.time()

def face_comparison(face_roi):
    # Thực hiện so sánh khuôn mặt với DeepFace
    # Replace 'test1.jpg' with the path to the image you want to compare with
    obj = DeepFace.verify(face_roi, "test6.png", enforce_detection = False)
    print(obj["verified"])

def main():
    global last_recognition_time
    while True:
        ret, frame = videoCapture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(30, 30),
        )

        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Lấy khuôn mặt từ frame
            face_roi = frame[y:y+height, x:x+width]

            # Kiểm tra và chạy DeepFace sau mỗi 5 giây
            current_time = time.time()
            if current_time - last_recognition_time >= 5:
                last_recognition_time = current_time
                face_comparison(face_roi)

        cv2.imshow('Face detection', frame)
        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    main()

videoCapture.release()
cv2.destroyAllWindows()
