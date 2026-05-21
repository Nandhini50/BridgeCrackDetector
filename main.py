import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

print("Program Started")

# STEP 1 - Read Image
img = cv2.imread('bridge.jpg')

if img is None:
    print("ERROR: Image not found")
    exit()

print("Image Loaded Successfully")

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display Original Image
plt.figure(figsize=(6,6))
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

# STEP 2 - Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(6,6))
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# STEP 3 - Median Filtering
filtered = cv2.medianBlur(gray, 3)

plt.figure(figsize=(6,6))
plt.imshow(filtered, cmap='gray')
plt.title("Median Filtered")
plt.axis('off')
plt.show()

# STEP 4 - CLAHE Contrast Enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(filtered)

plt.figure(figsize=(6,6))
plt.imshow(enhanced, cmap='gray')
plt.title("Enhanced Image")
plt.axis('off')
plt.show()

# STEP 5 - Canny Edge Detection
edges = cv2.Canny(enhanced, 100, 200)

plt.figure(figsize=(6,6))
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
plt.show()

# STEP 6 - Morphological Operations
kernel = np.ones((3,3), np.uint8)

dilated = cv2.dilate(edges, kernel, iterations=1)

closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(6,6))
plt.imshow(closed, cmap='gray')
plt.title("Morphological Closing")
plt.axis('off')
plt.show()

# STEP 7 - Remove Small Noise
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)

min_area = 50

output = np.zeros_like(closed)

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area >= min_area:
        output[labels == i] = 255

plt.figure(figsize=(6,6))
plt.imshow(output, cmap='gray')
plt.title("Noise Removed")
plt.axis('off')
plt.show()

# STEP 8 - Find Contours
contours, _ = cv2.findContours(output,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

result = img_rgb.copy()

# Draw rectangles around cracks
for contour in contours:

    x, y, w, h = cv2.boundingRect(contour)

    cv2.rectangle(result,
                  (x, y),
                  (x+w, y+h),
                  (255,0,0),
                  2)

# Final Result
plt.figure(figsize=(8,8))
plt.imshow(result)
plt.title("Detected Crack Regions")
plt.axis('off')
plt.show()

print("Crack Detection Completed")

input("Press Enter to Exit")