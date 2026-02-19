# pip install mediapipe==0.10.11 opencv-python, cv2-python
# run using python main.py
import cv2, time
import mediapipe as mp

# Initialize the model
mp_hands = mp.solutions.hands #OBJECT accesses hand tracking model
mp_drawing = mp.solutions.drawing_utils #OBJECT accesses draw utilities to draw
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) 

cap = cv2.VideoCapture(0) # OBJECT to capture video from webcam

print("Hand Tracker Started! Press 'q' to quit.")

# MAIN LOOP
while cap.isOpened():
    success, frame = cap.read() #success bool, frame is image
    frame = cv2.flip(frame, 1) # horizontal flip
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert from BGR to RGB as cv + mp use diff defaults
    results = hands.process(rgb_frame) #OBJECT 


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            indexFx = int(hand_landmarks.landmark[8].x *w)#normalized x coordinate of index finger tip
            indexFy = int(hand_landmarks.landmark[8].y *h) #normalized y coordinate of index
            middleFx = int(hand_landmarks.landmark[12].x *w) #normalized x coordinate of middle finger tip
            middleFy = int(hand_landmarks.landmark[12].y *h) #normalized y coordinate of middle
            
            cv2.circle(frame, (indexFx, indexFy), 10, (255, 0, 0), -1) 
            cv2.circle(frame, (middleFx, middleFy), 10, (0, 255, 0), -1) 
            
            
            # Print index finger tip (ID 8)
            cv2.putText(frame, f"Finger Tip indxF Y: {hand_landmarks.landmark[8].y:.2f}, X: {hand_landmarks.landmark[8].x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Finger Tip middleF Y: {hand_landmarks.landmark[12].y:.2f}, X: {hand_landmarks.landmark[12].x:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)   
    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #key listener interrupt
        break

cap.release()
cv2.destroyAllWindows()

"""             mp_drawing.draw_landmarks( 
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            ) """