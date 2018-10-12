#!/usr/bin/env python
# -*- encoding: UTF-8 -*-

import math
import numpy as np

def replace_value(data):

    """
    Cabin
    """
    # Split Cabin.
    cabin = data['Cabin'].copy()
    data['Deck'] = cabin.str.slice(0, 1)
    data['Room'] = cabin.str.slice(1, 5).str.extract('([0-9]+)', expand=False).astype('float')

    # Replace Deck to number.
    data['DeckNumber'] = np.nan
    data['DeckNumber'] = data['DeckNumber'].where(data['Deck'] != 'A', 8)
    data['DeckNumber'] = data['DeckNumber'].where(data['Deck'] != 'B', 7)
    data['DeckNumber'] = data['DeckNumber'].where(data['Deck'] != 'C', 6)
    data['DeckNumber'] = data['DeckNumber'].where(data['Deck'] != 'D', 5)
    data['DeckNumber'] = data['DeckNumber'].where(data['Deck'] != 'E', 4)
    data['DeckNumber'] = data['DeckNumber'].where(data['Deck'] != 'F', 3)
    data['DeckNumber'] = data['DeckNumber'].where(data['Deck'] != 'G', 2)
    data['DeckNumber'] = data['DeckNumber'].where(data['Deck'] != 'T', 1)

    """
    Sex
    """
    data['Sex'] = data['Sex'].where(data['Sex'] != 'male', 1)
    data['Sex'] = data['Sex'].where(data['Sex'] != 'female', 0)

    """
    Fill missing values.
    """
    ref = data.copy()
    for idx in data.index:
        if math.isnan(data.loc[idx, 'Age']):
            data.loc[idx, 'Age'] = random_select(ref, 'Age')
        if math.isnan(data.loc[idx, 'Fare']):
            data.loc[idx, 'Fare'] = random_select(ref, 'Fare')
        if math.isnan(data.loc[idx, 'DeckNumber']):
            data.loc[idx, 'DeckNumber'] = random_select(ref, 'DeckNumber')
        if math.isnan(data.loc[idx, 'Room']):
            data.loc[idx, 'Room'] = random_room_select(ref, 'Room', data.loc[idx, 'DeckNumber'])

    return data


def random_select(data, col):
    return data[col].dropna().sample(1).values[0]


def random_room_select(data, col, deck_value):
    ref = data.copy()
    ref = ref[ref['DeckNumber'] == deck_value]
    return ref[col].sample(1).values[0]
