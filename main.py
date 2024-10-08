import cv2
import matplotlib.pyplot as plt
import numpy as np
import highest_hand
import process
import loader

cardpath = 'test/1.png'
debug = 0

original_image = cv2.imread(cardpath)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

player_hand = []
table5 = []

height, width, _ = original_image_rgb.shape
middle = height // 2
top_half = original_image_rgb[:middle, :]
bottom_half = original_image_rgb[middle:, :]

# if debug:
plt.imshow(original_image_rgb)
plt.show()

for iTurn, original_image_rgb in enumerate([bottom_half, top_half]):

    imgResult = original_image_rgb.copy()
    imgResult2 = original_image_rgb.copy()

    thresh = process.get_thresh(imgResult)

    corners_list = process.find_corners_set(thresh)

    if debug:
        plt.imshow(thresh)
        plt.grid()
        plt.show()

    for i, corners in enumerate(corners_list):
        top_left = corners[0][0]
        bottom_left = corners[1][0]
        bottom_right = corners[2][0]
        top_right = corners[3][0]

        # print(f'top_left: {top_left}')
        # print(f'bottom_left: {bottom_left}')
        # print(f'bottom_right: {bottom_right}')
        # print(f'top_right: {top_right}\n')

    flatten_card_set = process.find_flatten_cards(imgResult2, corners_list)

    if debug:
        for img_output in flatten_card_set:
            plt.imshow(img_output)
            plt.show()

    cropped_images = process.get_corner_snip(flatten_card_set)
    ranksuit_list = []

    for i, (img, original) in enumerate(cropped_images):

        drawable = img.copy()
        d2 = original.copy()

        contours, _ = cv2.findContours(drawable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts_sort = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        cnts_sort = sorted(cnts_sort, key=lambda x: cv2.boundingRect(x)[1])

        if debug:
            plt.grid(True)
            plt.subplot(1, len(cropped_images), i + 1)
            plt.imshow(img)

        ranksuit = []
        for i, cnt in enumerate(cnts_sort):
            x, y, w, h = cv2.boundingRect(cnt)
            x2, y2 = x + w, y + h

            crop = d2[y:y2, x:x2]
            if (i == 0):  # rank: 70, 125
                crop = cv2.resize(crop, (70, 125), 0, 0)
            else:  # suit: 70, 100
                crop = cv2.resize(crop, (70, 100), 0, 0)
            # convert to bin image
            _, crop = cv2.threshold(crop, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            crop = cv2.bitwise_not(crop)

            ranksuit.append(crop)

        ranksuit_list.append(ranksuit)

    if debug:
        plt.show()

    black_img = np.zeros((120, 70))
    if debug:
        plt.figure(figsize=(12, 6))
    for i, ranksuit in enumerate(ranksuit_list):

        rank = black_img
        suit = black_img
        try:
            rank = ranksuit[0]
            suit = ranksuit[1]
        except:
            pass

        if debug:
            plt.subplot(len(ranksuit_list), 2, i * 2 + 1)
            plt.imshow(rank, 'gray')
            plt.subplot(len(ranksuit_list), 2, i * 2 + 2)
            plt.imshow(suit, 'gray')
    if debug:
        plt.show()

    train_ranks = loader.Loader.load_ranks('imgs/ranks')
    train_suits = loader.Loader.load_suits('imgs/suits')

    # print(train_ranks[0].img.shape)
    # print(train_suits[0].img.shape)
    if debug:
        for i, rank in enumerate(train_ranks):
            plt.subplot(1, len(train_ranks), i + 1)
            plt.axis('off')
            plt.imshow(rank.img, 'gray')
        plt.show()

        for i, suit in enumerate(train_suits):
            plt.subplot(1, len(train_suits), i + 1)
            plt.axis('off')
            plt.imshow(suit.img, 'gray')

        plt.show()

    for it in ranksuit_list:
        try:
            rank = it[0]
            suit = it[1]
        except:
            continue

        rs = process.template_matching(rank, suit, train_ranks, train_suits, show_plt=debug)
        if iTurn == 0:
            player_hand.append(rs)
        if iTurn == 1:
            table5.append(rs)

d = {
    11: 'Jack',
    12: 'Queen',
    13: 'King',
    14: 'Ace',
    'S': 'Spades',
    'C': 'Clubs',
    'D': 'Diamonds',
    'H': 'Hearts'
}

# 9-high Straight Flush
# player_hand = [(7, 'S'), (5, 'S')]
# table5 = [(14, 'H'), (8, 'S'), (6, 'S'), (4, 'S'), (9, 'S')]

# Four of a Kind, Aces
# player_hand = [(14, 'S'), (14, 'H')]
# table5 = [(14, 'D'), (14, 'C'), (8, 'S'), (7, 'H'), (2, 'D')]

# Full House, Kings over Aces
# player_hand = [(13, 'S'), (13, 'C')]
# table5 = [(13, 'D'), (14, 'C'), (14, 'S'), (7, 'H'), (2, 'D')]

# Ace-high Flush
# player_hand = [(14, 'S'), (12, 'S')]
# table5 = [(10, 'S'), (8, 'S'), (6, 'S'), (4, 'S'), (2, 'H')]

# Ace-high Straight
# player_hand = [(14, 'S'), (13, 'C')]
# table5 = [(12, 'D'), (11, 'C'), (10, 'S'), (3, 'H'), (2, 'D')]

# Three of a Kind, Kings
# player_hand = [(13, 'S'), (13, 'C')]
# table5 = [(13, 'D'), (10, 'C'), (8, 'S'), (4, 'H'), (2, 'D')]

# Two pairs, Kings and 10s
# player_hand = [(13, 'S'), (10, 'H')]
# table5 = [(13, 'D'), (10, 'C'), (8, 'S'), (7, 'H'), (2, 'D')]

# One pair, Kings
# player_hand = [(13, 'S'), (10, 'H')]
# table5 = [(13, 'D'), (8, 'C'), (7, 'S'), (6, 'H'), (2, 'D')]

# High card, Ace
# player_hand = [(14, 'S'), (9, 'H')]
# table5 = [(8, 'C'), (7, 'D'), (5, 'S'), (4, 'H'), (2, 'D')]

cards = player_hand + table5
# print(cards)
print(f"Your hand:\n{highest_hand.translate(d, player_hand)}")
print(f"Cards on the table:\n{highest_hand.translate(d, table5)}")
print(f"Your highest hand: {highest_hand.highest_hand(cards, d)}")
