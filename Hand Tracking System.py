# Import necessary libraries
import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize video capture from default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Initialize hand detector with specified detection confidence and maximum hands to track
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Enter an infinite loop to continuously process video frames
while True:
    # Capture a frame from the camera
    success, img = cap.read()

    # Use the hand detector to find hands in the captured frame
    hands, img = detector.findHands(img)

    # Check if hands are detected
    if hands:
        # Extract information from the first detected hand
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # Landmark list of the hand
        centerPoint1 = hand1["center"]  # Center point of the hand
        handType1 = hand1["type"]  # Type of hand (left or right)

        # Determine the status of fingers on the first hand
        fingers1 = detector.fingersUp(hand1)

        # Check if there are two detected hands
        if len(hands) == 2:
            # Extract information from the second detected hand
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # Landmark list of the hand
            bbox2 = hand2["bbox"]  # Bounding box of the hand
            centerPoint2 = hand2["center"]  # Center point of the hand
            handType2 = hand2["type"]  # Type of hand (left or right)

            # Determine the status of fingers on the second hand
            fingers2 = detector.fingersUp(hand2)

            # Calculate the distance between the centers of the two hands
            length, info, img = detector.findDistance(centerPoint1, centerPoint2, img)

    # Display the processed image with tracked hands and information
    cv2.imshow("Image", img)

    # Wait for a key event for 1 millisecond
    cv2.waitKey(1)
