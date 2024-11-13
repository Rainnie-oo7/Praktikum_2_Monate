#!/usr/bin/env python
# coding: utf-8

# # Python Hand Gesture Recognition (Checkpoint 2)

# This checkpoint is at the end of Objective 2.
# 
# At this point, your code should be able to get input from the camera and display it on the screen as if it were a mirror. It should also draw a rectangle where the user should put their hand.

# ### Header: Importing libraries and creating global variables

# In[9]:


import numpy as np
import cv2

# Hold the background frame for background subtraction.
background = None
# Hold the hand's PennFudanPed so all its details are in one place.
hand = None
# Variables to count how many frames have passed and to set the size of the window.
frames_elapsed = 0
FRAME_HEIGHT = 200
FRAME_WIDTH = 300
# Humans come in a ton of beautiful shades and colors.
# Try editing these if your program has trouble recognizing your skin tone.
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18


# ### HandData: A class to hold all the hand's details and flags

# In[10]:


class HandData:
    top = (0,0)
    bottom = (0,0)
    left = (0,0)
    right = (0,0)
    centerX = 0
    prevCenterX = 0
    isInFrame = False
    isWaving = False
    fingers = None
    
    def __init__(self, top, bottom, left, right, centerX):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.centerX = centerX
        self.prevCenterX = 0
        isInFrame = False
        isWaving = False
        
    def update(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right


# ### write_on_image(): Write info related to the hand gesture and outline the region of interest

# In[11]:


# Here we take the current frame, the number of frames elapsed, and how many fingers we've detected
# so we can print on the screen which gesture is happening (or if the camera is calibrating).
def write_on_image(frame):
    text = "Searching..."

    if frames_elapsed < CALIBRATION_TIME:
        text = "Calibrating..."
    elif hand == None or hand.isInFrame == False:
        text = "No hand detected"
    else:
        if hand.isWaving:
            text = "Waving"
        elif hand.fingers == 0:
            text = "Rock"
        elif hand.fingers == 1:
            text = "Pointing"
        elif hand.fingers == 2:
            text = "Scissors"
    
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,( 0 , 0 , 0 ),2,cv2.LINE_AA)
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,(255,255,255),1,cv2.LINE_AA)

    # Highlight the region of interest.
    cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (255,255,255), 2)


# ### Main function: Get input from camera and call functions to understand it

# In[12]:


# Our region of interest will be the top right part of the frame.
region_top = 0
region_bottom = int(2 * FRAME_HEIGHT / 3)
region_left = int(FRAME_WIDTH / 2)
region_right = FRAME_WIDTH

frames_elapsed = 0

capture = cv2.VideoCapture(0)

while (True):
    # Store the frame from the video capture and resize it to the window size.
    ret, frame = capture.read()
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    # Flip the frame over the vertical axis so that it works like a mirror, which is more intuitive to the user.
    frame = cv2.flip(frame, 1)
    
    # Write the action the hand is doing on the screen, and draw the region of interest.
    write_on_image(frame)
    # Show the previously captured frame.
    cv2.imshow("Camera Input", frame)
    frames_elapsed += 1
    # Check if user wants to exit.
    if (cv2.waitKey(1) & 0xFF == ord('x')):
        break

# When we exit the loop, we have to stop the capture too.
capture.release()
cv2.destroyAllWindows()

