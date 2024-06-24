import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


class ColorHelper:

    @staticmethod
    def gray2bin(img):
        return cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    @staticmethod
    def reverse(img):
        return cv2.bitwise_not(img)


class MathHelper:
    @staticmethod
    def length(x1, y1, x2, y2):

        length = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))

        return length


def get_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 45, 90)
    kernel = np.ones((2, 2))
    dial = cv2.dilate(canny, kernel=kernel, iterations=2)

    return dial


def find_corners_set(img):
    # find the set of contours on the threshold image
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # sort them by highest area
    proper = sorted(contours, key=cv2.contourArea, reverse=True)

    four_corners_set = []

    for cnt in proper:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)

        # only select those contours with a good area
        if area > 10000:
            # find out the number of corners
            approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, closed=True)
            num_corners = len(approx)

            if num_corners == 4:

                # make sure the image is oriented right: top left, bot left, bot right, top right
                l1 = np.array(approx[0]).tolist()
                l2 = np.array(approx[1]).tolist()
                l3 = np.array(approx[2]).tolist()
                l4 = np.array(approx[3]).tolist()

                finalOrder = []

                # sort by X value
                sortedX = sorted([l1, l2, l3, l4], key=lambda x: x[0][0])

                # sortedX[0] and sortedX[1] are the left half
                finalOrder.extend(sorted(sortedX[0:2], key=lambda x: x[0][1]))

                # now sortedX[1] and sortedX[2] are the right half
                # the one with the larger y value goes first
                finalOrder.extend(sorted(sortedX[2:4], key=lambda x: x[0][1], reverse=True))

                four_corners_set.append(finalOrder)

    return four_corners_set


def find_flatten_cards(img, set_of_corners):
    #desired width and height for the output card images
    width, height = 200, 300
    img_outputs = []

    for i, corners in enumerate(set_of_corners):
        top_left = corners[0][0]
        bottom_left = corners[1][0]
        bottom_right = corners[2][0]
        top_right = corners[3][0]

        # get the 4 corners of the card
        pts1 = np.float32([top_left, bottom_left, bottom_right, top_right])
        # now define which corner we are referring to
        pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

        # transformation matrix
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        img_output = cv2.warpPerspective(img, matrix, (width, height))

        img_outputs.append(img_output)

    return img_outputs


def get_corner_snip(flattened_images: list):
    corner_images = []
    for img in flattened_images:
        # crop the image to where the corner might be
        # vertical, horizontal
        crop = img[5:110, 1:40]

        # resize by a factor of 4
        crop = cv2.resize(crop, None, fx=4, fy=4)

        # threshold the corner
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bin_img = ColorHelper.gray2bin(gray)
        bilateral = cv2.bilateralFilter(bin_img, 11, 174, 17)
        canny = cv2.Canny(bilateral, 40, 24)
        kernel = np.ones((1, 1))
        result = cv2.dilate(canny, kernel=kernel, iterations=2)

        # append the thresholded image and the original one
        corner_images.append([result, bin_img])

    return corner_images


def template_matching(rank, suit, train_ranks, train_suits, show_plt=False) -> tuple[int, str]:

    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "0"
    best_suit_match_name = "Unknown"

    for train_rank in train_ranks:

        diff_img = cv2.absdiff(rank, train_rank.img)

        rank_diff = int(np.sum(diff_img) / 255)

        if rank_diff < best_rank_match_diff:
            best_rank_match_diff = rank_diff
            best_rank_name = train_rank.name

            if show_plt:
                print(f'diff score: {rank_diff}')
                plt.subplot(1, 2, 1)
                plt.imshow(diff_img, 'gray')

        plt.show()

    for train_suit in train_suits:

        diff_img = cv2.absdiff(suit, train_suit.img)
        suit_diff = int(np.sum(diff_img) / 255)

        if suit_diff < best_suit_match_diff:
            best_suit_match_diff = suit_diff
            best_suit_name = train_suit.name

            if show_plt:
                print(f'diff score: {suit_diff}')
                plt.subplot(1, 2, 2)
                plt.imshow(diff_img, 'gray')

        plt.show()

    if best_rank_match_diff < 2300:
        best_rank_match_name = best_rank_name

    if best_suit_match_diff < 1000:
        best_suit_match_name = best_suit_name

    plt.show()

    return int(best_rank_match_name), best_suit_match_name
