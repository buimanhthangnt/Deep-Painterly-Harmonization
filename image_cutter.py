import cv2
import imutils
import numpy as np
import pickle


points = []
global clicked
clicked = False

def onmouse(event, x, y, flags, param):
    global clicked
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        clicked = True
    elif event == cv2.EVENT_MOUSEMOVE and clicked:
        points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        clicked = False
        points.append((x, y))

style_img = cv2.imread('data/style.jpg')
style_img = cv2.resize(style_img, (800, 800))

target_img = cv2.imread('data/1.jpg')
target_img = imutils.resize(target_img, width=800)[50:,0:]

cv2.namedWindow('test')
cv2.setMouseCallback('test', onmouse)

cv2.imshow('test', target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

mask = np.zeros(target_img.shape)
cv2.drawContours(mask, [np.array(points)], -1, (0, 255, 0), -1)

scale = 0.5
mask = cv2.resize(mask, (int(mask.shape[1] * scale), int(mask.shape[0] * scale)))
target_img = cv2.resize(target_img, (int(target_img.shape[1] * scale), int(target_img.shape[0] * scale)))

background, obj = [], []
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i,j,1] != 255:
            background.append((i,j))
        else:
            obj.append((i,j))

# background = pickle.load(open('background.pkl', 'rb'))
# obj = pickle.load(open('obj.pkl', 'rb'))

pickle.dump(background, open('background.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
pickle.dump(obj, open('obj.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

# Create style
cv2.imwrite('data/style.jpg', style_img)

tx, ty = 380, 350
# Create naive
naive = style_img.copy()
for (i,j) in obj:
    naive[i+ty,j+tx,:] = target_img[i,j,:]
cv2.imwrite('data/naive.jpg', naive)

# Create mask
mask = np.zeros_like(style_img)
for (i,j) in obj:
    mask[i+ty,j+tx,:] = 255
cv2.imwrite('data/mask.jpg', mask)

# Create mask dilated
mask = cv2.dilate(mask, np.ones((10, 10),np.uint8), 10)
cv2.imwrite('data/mask_dilated.jpg', mask)
