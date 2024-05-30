import cv2

right_cap = cv2.VideoCapture(0)
left_cap = cv2.VideoCapture(1)

num = 0

while right_cap.isOpened():

    succes1, right_img = right_cap.read()
    succes2, left_img = left_cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/stereoRight/imageR' + str(num) + '.png', right_img)
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', left_img)
        print("images saved!")
        num += 1

    cv2.imshow('Right Image',right_img)
    cv2.imshow('Left Image',left_img)

# Release and destroy all windows before termination
right_cap.release()
left_cap.release()

cv2.destroyAllWindows()
