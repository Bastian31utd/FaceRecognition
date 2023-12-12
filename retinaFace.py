from retinaface import RetinaFace
from deepface import DeepFace
import cv2

# Load model from RetinaFace
retinaface = RetinaFace

# Đường dẫn đến ảnh cần xử lý
input_image_path = "/Users/admin/IdeaProjects/FaceRecognition/test2.jpg"
output_image_path = "/Users/admin/IdeaProjects/FaceRecognition/test2.jpg"

# Đọc ảnh đầu vào
frame = cv2.imread(input_image_path)

# Phát hiện khuôn mặt trong ảnh
result = retinaface.detect_faces(frame)

for face in result.keys():
    identity = result[face]
    area = identity["facial_area"]
    
    # Lấy tọa độ khuôn mặt
    x1, y1, x2, y2 = area[0], area[1], area[2], area[3]
    
    # Cắt khuôn mặt từ ảnh gốc
    face_image = frame[y1:y2, x1:x2]

    # Lưu ảnh khuôn mặt đã cắt
    cv2.imwrite(output_image_path, face_image)

    # Hiển thị ảnh đã cắt khuôn mặt
    cv2.imshow('Cropped Face', face_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
