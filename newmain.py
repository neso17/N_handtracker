# pip install mediapipe==0.10.11 opencv-python, cv2-python
# run using python main.py
import cv2
import mediapipe as mp

# Access sub-modules solution through  mp object
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#BaseOptions = mp.tasks.BaseOptions
#GestureRecognizer = mp.tasks.vision.GestureRecognizer
#GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
#GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
#VisionRunningMode = mp.tasks.vision.RunningMode

#options = GestureRecognizerOptions(
#    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
#    running_mode=VisionRunningMode.LIVE_STREAM,
#    result_callback=print_result)
#with GestureRecognizer.create_from_options(options) as recognizer:
# Initialize  model
     
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print("Hand Tracker Started! Press 'q' to quit.")
count = 0

while cap.isOpened() and count < 3:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
            
            # Print index finger tip (ID 8)
            print(f"DaddyF Tip Y: {hand_landmarks.landmark[4].y:.2f}")
            print(f"MummyF Tip Y: {hand_landmarks.landmark[8].y:.2f}")
            print(f"BrotherF Tip Y: {hand_landmarks.landmark[12].y:.2f}")
            print(f"SisterF Tip Y: {hand_landmarks.landmark[16].y:.2f}")
            print(f"BabyF Tip Y: {hand_landmarks.landmark[20].y:.2f}")
            print(f"")

        count += 1

    cv2.imshow("Hand Tracker", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#def print_result(result: GestureRecogniserResult, ouput_image: mp.Image, timestamp_ms: int):
 #   print('result: {}'.format(result))

cap.release()
cv2.destroyAllWindows()