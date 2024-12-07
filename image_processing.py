import cv2
import numpy as np
import matplotlib.pyplot as plt

# Yol fayllarını daxil edin
config_path = "C:/Users/ASUS/Desktop/YOLO/yolov3.cfg"  # Konfiqurasiya faylı
weights_path = "C:/Users/ASUS/Desktop/YOLO/yolov3.weights"  # Çəkilər faylı
image_path = "C:/Users/ASUS/Desktop/YOLO/city.jpg"  # Təsvir faylı

# Modeli yükləyin
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Sinif adlarını fayldan oxuyun (adlar faylı daxil edilməlidir)
names_path = "C:/Users/ASUS/Desktop/YOLO/coco.names"  # Sinif adları faylı
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Test üçün təsviri oxuyun
image = cv2.imread(image_path)
height, width, _ = image.shape

# Təsviri YOLO üçün çevrilmiş formata salın
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Modelin çıxış qatlarını əldə edin
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Tanıma əməliyyatı aparın
detections = net.forward(output_layers)

# Tanınmış obyektləri və koordinatları saxlamaq
boxes = []
confidences = []
class_ids = []

for output in detections:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence > 0.5:  # Yüksək inamlı obyektləri saxla
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # Koordinatları təyin et
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Obyektləri təkrarlama ilə çıxartmaq
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Obyektləri təsvir üzərində çək
for i in indices.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    color = (0, 255, 0)  # Yaşıl çərçivə
    
    # Çərçivə və sinif adı təsvirə əlavə et
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Təsviri göstərin və yadda saxlayın
cv2.imshow("Detected Objects", image)
cv2.imwrite("output_image.jpg", image)

# Göstərilən pəncərəni bağlamaq üçün düymə sıxın
cv2.waitKey(0)
cv2.destroyAllWindows()
