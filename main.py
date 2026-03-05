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

prev_pos = {"x": 0, "y": 0} 

# MAIN LOOP
while cap.isOpened():
    time.sleep(0.67)
    success, frame = cap.read() #success bool, frame is image
    frame = cv2.flip(frame, 1) # horizontal flip
    h, w, _ = frame.shape #error!
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert from BGR to RGB as cv + mp use diff defaults
    results = hands.process(rgb_frame) #OBJECT 
    
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            curr_pos = {             # dict for current frame
                "x": hand_landmarks.landmark[8].x,
                "y": hand_landmarks.landmark[8].y
            }
            x_coord = int(curr_pos["x"] * w)
            y_coord = int(curr_pos["y"] * h)
            dx = curr_pos["x"] - prev_pos["x"]
            dy = curr_pos["y"] - prev_pos["y"]

            cv2.circle(frame, (x_coord, y_coord), 10, (255, 0, 0), -1)             
            cv2.putText(frame, f"Finger Tip indxF Y: {hand_landmarks.landmark[8].y:.2f}, X: {hand_landmarks.landmark[8].x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            print(dx)
            if abs(dx) > 0.1:
                if dx > 0:
                    gesture = "Swipe Right"
                elif dx < 0:
                    gesture = "Swipe Left"
            elif abs(dy) > 0.1:
                if dy > 0:
                    gesture = "Swipe Down" 
                elif dy < 0:
                    gesture = "Swipe Up"

            else:
                    gesture = "No Swipe"

            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 0, 255), 2)
            prev_pos = curr_pos
    
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


