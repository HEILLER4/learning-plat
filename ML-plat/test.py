import cv2

img = cv2.imread("cat.png")
def draw_cri(event, x, y, flags, param):
   font = cv2.FONT_HERSHEY_SIMPLEX
   
   if event == cv2.EVENT_LBUTTONDOWN:
      b, g, r = img[y, x]
      print(f"pixel {x}, {y} bgr: {b}, {g}, {r}")
  
      cv2.circle(img, (x,y), 50, (255, 255, 255), -1)
      cv2.imshow("image", img)
   elif event == cv2.EVENT_RBUTTONDOWN:
      b, g, r = img[y, x]
      cv2.putText(img, f"{b},{g},{r}", (x,y), font, 2, (255,255,255), 5, cv2.LINE_AA)
      cv2.imshow("image", img) 
cv2.imshow("image", img)
cv2.setMouseCallback("image", draw_cri)


cv2.waitKey(0)
cv2.destroyAllWindows()
