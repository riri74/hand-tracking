import cv2 as cv
import mediapipe as mp
import time # to check the frame rate

# enable camera from cv
cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: #extract info on each hand ; hand lms is individual hand
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 0:
                    cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (10, 17), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


    cv.imshow('image',img) #displays the window with name image
    cv.waitKey(1) # waits fr 1ms for a key press on the window
