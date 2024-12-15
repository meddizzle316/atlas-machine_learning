#!/usr/bin/env python3
"""takes in user input and gives a basic response"""


while True:
    question = input("Q: ")

    question_lower = question.lower()
    if question_lower in ['exit', 'quit', 'goodbye', 'bye']:
        print("A: Goodbye")
        break
    else:
        print("A: ")