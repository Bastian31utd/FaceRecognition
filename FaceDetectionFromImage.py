import cv2

imagePath = "test5.jpg"
checkPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(checkPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor = 1.2,
    minNeighbors = 4,
    minSize = (30, 30)
)

print("Found {0} faces!".format(len(faces)))

# Draw rectangles around faces
# (x, y): left-above
# (0, 255, 0): color of the rectangle border
# 2: thickness of the rectangle border
for (x, y, width, height) in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

image_height, image_width, _ = image.shape
cv2.namedWindow('Faces found', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Faces found', image_width, image_height)

# Display the image with detected faces
cv2.imshow("Faces found", image)
cv2.waitKey(0)