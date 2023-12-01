import cv2

videoCapture = cv2.VideoCapture(0)
if not videoCapture.isOpened():
    print("Can not connect to camera")
    exit()

checkPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(checkPath)

while True:
    # Capture frame-by-frame
    # ret: boolean yes or no
    # frame: frame
    ret, frame = videoCapture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 4,
        minSize = (30, 30),
    )

    # Draw a rectangle around the faces
    # (x, y): left-above
    # (0, 255, 0): color of the rectangle border
    # 2: thickness of the rectangle border
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display the frames with detected faces
    cv2.imshow('Face detection', frame)

    # ESC to exit
    if cv2.waitKey(1) == 27:
        break

# print("Found {0} faces!".format(len(faces)))
videoCapture.release()
cv2.destroyAllWindows()
