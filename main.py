#!/usr/bin/env python3

# File for organising Configuration

from AI import ai

# m = ai(iterations = 3, amnt_of_examples = 20)
# m.train()
# m.save()

m = ai()
m.load("CSXO")
m.test(10)
