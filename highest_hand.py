def translate(dictonary, cards):
    string = ""
    for item in cards:
        string += f"{dictonary.get(item[0], str(item[0]))} of {dictonary.get(item[1])}\n"
    return string


def highest_hand(cards, d):
    ranks_count = {}
    suits_count = {}
    ranks = set()
    cards = sorted(cards, key=lambda x: x[0], reverse=True)
    for card in cards:
        ranks.add(card[0])
        if card[0] in ranks_count:
            ranks_count[card[0]] += 1
        else:
            ranks_count[card[0]] = 1

        if card[1] in suits_count:
            suits_count[card[1]] += 1
        else:
            suits_count[card[1]] = 1

    ranks_count = sorted(ranks_count.items(), key=lambda x: x[1], reverse=True)
    suits_count = sorted(suits_count.items(), key=lambda x: x[1], reverse=True)
    ranks = sorted(ranks, reverse=True)

    is_straight = False
    straight = [ranks[0]]
    last = ranks[0]
    for val in ranks:
        if last - val == 1:
            straight.append(val)
        else:
            straight = [val]
        last = val
        if len(straight) == 5:
            is_straight = True
            break

    is_flush = (suits_count[0][1] >= 5)

    # straight flush
    if is_straight and is_flush:
        tmp = []
        for card in cards:
            if card[1] == suits_count[0][0]:
                tmp.append(card[0])
        if set(straight).issubset(tmp):
            return f"{d.get(straight[0], str(straight[0]))}-high Straight Flush"
    # four of a kind
    if ranks_count[0][1] == 4:
        return f"Four of a Kind, {d.get(ranks_count[0][0], str(ranks_count[0][0]))}s"
    # full house
    if ranks_count[0][1] == 3 and ranks_count[1][1] == 2:
        return f"Full House, {d.get(ranks_count[0][0], str(ranks_count[0][0]))}s over {d.get(ranks_count[1][0], str(ranks_count[1][0]))}s"
    # flush
    if is_flush:
        for card in cards:
            if card[1] == suits_count[0][0]:
                return f"{d.get(card[0], str(card[1]))}-high Flush"
    # straight
    if is_straight:
        return f"{d.get(straight[0], str(straight[0]))}-high Straight"
    # three of a kind
    if ranks_count[0][1] == 3:
        return f"Three of a Kind, {d.get(ranks_count[0][0], str(ranks_count[0][0]))}s"
    # two pairs
    if ranks_count[0][1] == 2 and ranks_count[1][1] == 2:
        return f"Two pairs, {d.get(ranks_count[0][0], str(ranks_count[0][0]))}s and {d.get(ranks_count[1][0], str(ranks_count[1][0]))}s"
    # one pair
    if ranks_count[0][1] == 2:
        return f"One pair, {d.get(ranks_count[0][0], str(ranks_count[0][0]))}s"
    else:
    # high card
        return f"High card, {d.get(ranks[0], str(ranks[0]))}"
    
