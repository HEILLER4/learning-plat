import cv2

img = cv2.imread("cat.png")

# Outer rectangle (filled)
cv2.rectangle(img, (10, 10), (200, 200), (0, 255, 0), -1)

# Inner rectangle (filled)
cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)

cv2.imshow("Rectangles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
