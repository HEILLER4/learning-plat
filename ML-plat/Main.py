import cv2

img = cv2.imread("cat.png")

def show_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        display = img.copy()
        b, g, r = img[y, x]

        text = f"{r},{g},{b}"
        cv2.putText(display, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 1)

        cv2.imshow("Image", display)

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", show_pixel)

cv2.waitKey(0)
cv2.destroyAllWindows()
