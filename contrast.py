import cv2
import numpy as np

img = cv2.imread('3l9eq5e6a4o81.webp', 1)

lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_channel, a, b = cv2.split(lab)


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl = clahe.apply(l_channel)


limg = cv2.merge((cl,a,b))


enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

result = np.hstack((img, enhanced_img))
cv2.imshow('Result', result)
cv2.waitKey()