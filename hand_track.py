import cv2
import mediapipe as mp
import time
import handTrackingModule as htm


p_time = 0
c_time = 0

cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_tip = detector.find_position(img)
    if len(lm_tip) != 0:
        print(lm_tip[20])
    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("video", img)
    cv2.waitKey(1)
