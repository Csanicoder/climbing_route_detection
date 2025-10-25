import cv2
import numpy as np

img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.line(img, (0, 0), (200, 200), (0, 0, 255), 3)
cv2.imshow("Line", img)
cv2.waitKey(0)