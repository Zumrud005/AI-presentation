import cv2
import matplotlib.pyplot as plt

# Təsvirin tam yolunu qeyd edin
image_path = "C:/Users/ASUS/Desktop/71UD0YtGx7L._AC_UF1000,1000_QL80_.jpg"

# Təsviri yükləyirik
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Faylın yüklənib-yüklənmədiyini yoxlayırıq
if image is None:
    print("Təsvir faylı yüklənə bilmədi. Fayl yolunu yoxlayın!")
    exit()

# Gaussian Blur tətbiq edirik
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Canny Edge Detection tətbiq edirik
edges = cv2.Canny(image, threshold1=100, threshold2=200)

# Nəticələri göstəririk
plt.figure(figsize=(10, 5))

# Əsas təsvir
plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Əsas Təsvir")
plt.axis("off")

# Gaussian Blur təsviri
plt.subplot(1, 3, 2)
plt.imshow(blurred_image, cmap="gray")
plt.title("Bulanıq Təsvir")
plt.axis("off")

# Kənarların aşkar edilməsi
plt.subplot(1, 3, 3)
plt.imshow(edges, cmap="gray")
plt.title("Kənarlar")
plt.axis("off")

plt.tight_layout()
plt.show()

