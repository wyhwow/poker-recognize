############## Python-OpenCV Playing Card Detector ###############
#
# Author: Evan Juras
# Date: 9/5/17
# Description: Python script to detect and identify playing cards
# from a
#  video feed.
#

# Import necessary packages
import cv2
import numpy as np
import time
import os
import Cards


### ---- INITIALIZATION ---- ###
# Define constants and initialize variables

WIDTH = 1080

## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
train_suits = Cards.load_suits(path + '/Card_Imgs/')

### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.

cam_quit = 0  # Loop control variable

# Begin capturing frames
if cam_quit == 0:

    # Grab frame from video stream
    image = cv2.imread('./uploaded_files/real.jpg')
    w0,w1,w2 = image.shape
    # print(w0,w1)
    if w0 > w1:
        ratio = float(w1)/WIDTH
        # print(ratio)
        w1 = WIDTH
        w0 = int(w0/ratio)
    else:
        ratio = float(w0)/WIDTH
        # print(ratio)
        w0 = WIDTH
        w1 = int(w1/ratio)
    image = cv2.resize(image, (w1,w0))
    # image = cv2.medianBlur(image,3)

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Pre-process camera image (gray, blur, and threshold it)
    max = 0
    cnts_sort = []
    cnts_is_card = []

    for thresh in range(150,50,-10):
        pre_proc = Cards.preprocess_image(image,thresh)
    # Find and sort the contours of all cards in the image (query cards)
        cnts_sort_tmp, cnt_is_card_tmp = Cards.find_cards(pre_proc)
        card_num = cnt_is_card_tmp.tolist().count(1)
        # print("Get count", card_num, "with thresh:", thresh)
        if (card_num) > max:
            max = card_num
            cnts_sort = cnts_sort_tmp
            cnt_is_card = cnt_is_card_tmp
        # if (card_num) < max:
        #     break

    # If there are no contours, do nothing
    if len(cnts_sort) != 0:

        # Initialize a new "cards" list to assign the card objects.
        # k indexes the newly made array of cards.
        cards = []
        k = 0

        # For each contour detected:
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):
                # Create a card object from the contour and append it to the list of cards.
                # preprocess_card function takes the card contour and contour and
                # determines the cards properties (corner points, etc). It generates a
                # flattened 200x300 image of the card, and isolates the card's
                # suit and rank from the image.
                cards.append(Cards.preprocess_card(cnts_sort[i], image))

                # Find the best rank and suit match for the card.
                cards[k].best_rank_match, cards[k].best_suit_match, cards[k].rank_diff, cards[
                    k].suit_diff = Cards.match_card(cards[k], train_ranks, train_suits)

                # Draw center point and match result on the image.
                k = k + 1
        image = Cards.draw_results(image, cards)

        # Draw card contours on image (have to do contours all at once or
        # they do not show up properly for some reason)
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)

    # Draw framerate in the corner of the image. Framerate is calculated at the end of the main loop,
    # so the first time this runs, framerate will be shown as 0.

    # Finally, display the image with the identified cards!
    # Calculate framerate
    t2 = cv2.getTickCount()
    time = (t2-t1)/cv2.getTickFrequency()
    # print('Time= '+str(time))

    # cv2.imshow("Card Detector", image)
    # Poll the keyboard. If 'q' is pressed, exit the main loop.
    key = cv2.waitKey(0) & 0xFF
    # if key == ord("q"):
    #     cam_quit = 1
    cv2.imwrite('./static/images/result.jpg',image)

cv2.destroyAllWindows()

