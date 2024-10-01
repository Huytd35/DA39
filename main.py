import cv2
import numpy as np

# Load model MobileNet SSD
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Lớp đối tượng mà mô hình có thể phát hiện
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Mở hình ảnh
image = cv2.imread('test.jpg')
(h, w) = image.shape[:2]

# Tạo blob từ hình ảnh để đưa vào mạng
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# Đưa blob vào mạng
net.setInput(blob)
detections = net.forward()

# Lập danh sách để lưu tọa độ của xe
cars = []

# Duyệt qua các phát hiện
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    # Chỉ xử lý các phát hiện có độ tin cậy cao
    if confidence > 0.5:
        idx = int(detections[0, 0, i, 1])

        # Kiểm tra nếu đối tượng được phát hiện là ô tô (class 'car')
        if CLASSES[idx] == "car":
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cars.append((startX, startY, endX, endY))

            # Vẽ khung chữ nhật xung quanh các ô tô
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Hiển thị hình ảnh với các khung hình ô tô
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Kiểm tra ô đậu xe có đầy hay không
parking_slots = [(50, 50, 200, 200), (250, 50, 400, 200)]  # Giả sử có 2 ô đậu xe
for slot in parking_slots:
    slot_full = False
    for car in cars:
        if (car[0] >= slot[0] and car[2] <= slot[2] and
            car[1] >= slot[1] and car[3] <= slot[3]):
            slot_full = True
            break
    if slot_full:
        print(f"Slot {slot} is full.")
    else:
        print(f"Slot {slot} is empty.")
