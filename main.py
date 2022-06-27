#!/usr/bin/env python3

# File for organising Configuration

from AI import ai

# m = ai(iterations = 750, amnt_of_examples = 750)
# m.train()
# m.save()

m = ai()
m.load("PJVC")
m.test(10)

# from AI import MakeCSVFromRaw

# MakeCSVFromRaw()