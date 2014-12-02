#! /usr/bin/python
#Random Game of Rock Paper Scissors 

###### Try to stay consistent with terminology
###### RPS = rock, paper, scissors
###### match = 1 game of RPS
###### round = a predetermined amount of matches

import random

import os

import sys

import pybrain           # For AI implementation
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import GaussianLayer, SoftmaxLayer, FeedForwardNetwork, LinearLayer, SigmoidLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer

import gaussian

import numpy

import math
from pybrain.utilities import percentError



round_len = 10       	 # 10 matches in 1 round
inputdim = round_len-1
hiddenneuron = 1      
outputdim = 3
momentum = 0.1
verbose = True
weightdecay = .01
inLayer = LinearLayer(round_len)
hiddenLayer = GaussianLayer(3)
outLayer = LinearLayer(1)


ds = ClassificationDataSet(inputdim, 1, nb_classes=3, class_labels=['rock', 'paper', 'scissors'])
myNetwork = buildNetwork(inputdim, hiddenneuron, outputdim, outclass=GaussianLayer )

p_play_history = []
score = [0, 0, 0]


def clear():        	 # Clear screen
    os.system("clear")


def get_rounds():        # Get the number of rounds
    nr_rounds = raw_input("How many rounds would you like to play (must be an odd number): ")
    nr_rounds = int(nr_rounds)
    while nr_rounds%2 == 0 or nr_rounds <= -1:
        print "Pick an ODD number!"
        nr_rounds = raw_input("Let's try this again: ")
        nr_rounds = int(nr_rounds)
    print "OK, let's play some rounds!", nr_rounds
    return nr_rounds


def determine_winner(p_choice, c_choice, score):
    if (p_choice == c_choice):                  # Tie
        print "TIE", score[2]+1
        score[2] = score[2]+1
    elif (int(p_choice)+2 == int(c_choice)):    # Player Won
        print "You WON!!!"
        score[0] = score[0]+1
    else:                                       # Player Lost
        print "You LOST."
        score[1] = score[1]+1


def training (myNetwork, ds):
    print "\n===========Training the data..."
    trnData, tstData = ds.splitWithProportion(0.5)		## trnData will have % indicated here
    trnData._convertToOneOfMany()
    tstData._convertToOneOfMany()
    
    print "Number of training pattern: ", len(trnData)
    len_trnData = len(trnData)
    if (len(trnData) and len(tstData)):
        trainer = BackpropTrainer(myNetwork, dataset=trnData, momentum = .05, verbose = True, weightdecay = .01)
        rpTrainer = RPropMinusTrainer(myNetwork, verbose=True)
        for i in range (5):
            print "\n"
            rpTrainer.trainOnDataset(dataset=trnData)
	
    return


def printResults(score):
    print "\n\n Current results (win, lose, tie): "
    print score

 
def init_5_rounds(p_play_history, score):
    # To get an idea of the player's playing style
    decision = (1, 2, 3)
    global ds

    for i in range(0,round_len):
        ai = random.choice(decision)
        player = ''	
        match_nr = i+1
        while player != 'r' and player != 'p' and player != 's':
            print "\nROUND ", match_nr    
            player = raw_input('What is your choice? (r)ock, (p)aper, (s)cissors: ')
	    
        if player == 'r':
            p_rock    = 1
            p_paper   = 0
            p_scissors = 0
            player = 1
            match = [p_rock, p_paper, p_scissors]
            
        elif player == 'p':              # Paper = 2
            p_rock    = 0
            p_paper   = 1
            p_scissors = 0
            player = 2
            match = [p_rock, p_paper, p_scissors]

        elif player == 's':              # Scissors = 3
            p_rock    = 0
            p_paper   = 0
            p_scissors = 1 
            player = 3
            match = [p_rock, p_paper, p_scissors]       
       
        #print "player = ", player
        p_play_history.append(player)

        print "====================="
        #print "p_play_history: ", p_play_history
		    
        #if match[0] == 1:                # Rock = 1
            #print "You played: ROCK" 
        #elif match[1] == 1:              # Paper = 2
            #print "You played: PAPER"
        #elif match[2] == 1:              # Scissors = 3
            #print "You played: SCISSORS"

        if ai == 1:
            print "I played: ROCK\n"
        elif ai == 2:
            print "I played: PAPER\n"
        elif ai == 3:
            print "I played: SCISSORS\n"

        determine_winner(player, ai, score)
        print "==================="
        #print "player, ai", player, ai
        #print "Score = ", score
        
        if (match_nr == round_len):
            prev_match_nr = match_nr-1
            #print prev_match_nr

            p_input = p_play_history[0:prev_match_nr]
            #print "prev_round input= ", p_input
            #print "length of prev_input", len(p_input)

            output = p_play_history[match_nr-1]
            #print "output = ", output
            
            if (output == [1,0,0]):
                print "rock"
                output = 1
            if (output == [0,1,0]):
                print "paper"
                output = 2
            if (output == [0,0,1]):
                print "scissors"
                output = 3
            
            ds.addSample(p_input, [player], )
            #print "ds = = = ", ds
    return myNetwork


def turn(p_play_history, nr_matches, score, myNetwork):                 # Initialization of round
    global ds
  
    for i in range(nr_matches):
        if (i % round_len == 0):
            training(myNetwork, ds)
            print"\n\nCurrent Score:"
            print score
        decision = ('1', '2', '3')
        player = ''
        match_nr = i+round_len+1
        #print "match_nr = ",match_nr

        while player != 'r' and player != 'p' and player != 's':
            print "\n********** ROUND ", match_nr
            player = raw_input('What is your choice? (r)ock, (p)aper, (s)cissors: ')

        someData = ClassificationDataSet(inputdim, 1, nb_classes=3, class_labels = ['1', '2', '3'])

        recent_sample = p_play_history[match_nr-round_len:match_nr-1]
        #print "\tRecent Sample", recent_sample
        prediction = someData.addSample(recent_sample, [0])

        prediction = myNetwork.activate(recent_sample)
        numpy.argmax(prediction) 
        #print "Prediction = ", prediction
        prediction = prediction.argmax()
        ai = prediction
        #print "MATCH PREDICTION: ", prediction

        if ai  == 1:
            ai = 2
        elif ai ==2:
            ai = 3
        elif ai == 3:
            ai = 1

        #print "AI going to play: ", ai 
        if player == 'r':                # Rock = 1
            p_rock    = 1
            p_paper   = 0
            p_scissors = 0
            player = 1
            match  = (p_rock, p_paper, p_scissors)
            p_play_history.append(player)

        elif player == 'p':              # Paper = 2
            p_rock    = 0
            p_paper   = 1
            p_scissors = 0
            player = 2
            match = (p_rock, p_paper, p_scissors)
            p_play_history.append(player)

        elif player == 's':              # Scissors = 3
            p_rock    = 0
            p_paper   = 0
            p_scissors = 1 
            player = 3
            match = (p_rock, p_paper, p_scissors)
            p_play_history.append(player)
            
    
        print "====================="

        if p_rock == 1:                # Rock = 1
            print "You played: ROCK" 
        elif p_paper == 1:              # Paper = 2
            print "You played: PAPER"
        elif p_scissors == 1:              # Scissors = 3
            print "You played: SCISSORS"

        if ai == 1:
            print "I played: ROCK\n"
        elif ai == 2:
            print "I played: PAPER\n"
        elif ai == 3:
            print "I played: SCISSORS\n"
        
        determine_winner(player, ai, score)
        print "==================="
        
        prev_match_nr = match_nr-1
        #print "Previous match number = ", prev_match_nr
    	prev_round = p_play_history[match_nr-round_len:prev_match_nr]
        #print "previous round plays: ", prev_round
    	#print "previous round len: ", len(prev_round)
    	ds.addSample(prev_round, [player])
    	#print "New dataset: ",ds


########## BEGIN PLAYING #####################################
clear()
print "Welcome to ROCK PAPER SCISSORS!  I will predict your next move and play accordingly.\nFirst let's play 10 rounds so I can get an idea of your playing style..\n\n"
myNetwork = init_5_rounds(p_play_history, score)
printResults(score)
nr_matches = get_rounds()
print "\nNow I will start trying to predict your next play\n"
turn(p_play_history, nr_matches, score, myNetwork)
########## RESULTS ######################################
print "\nYour final Score (win, lost, tie):"
print score
print "\n\nThanks for playing!\n\n\n"
