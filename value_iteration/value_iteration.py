#Benjamin Stanelle    1001534907
import sys
from sys import exit
import os
import numpy as np
import copy

def value_iteration():
#===============Command Line Arguments======================================
    environment_file=sys.argv[1] #is the path name of a file that describes the environment where the agent moves
    non_terminal_reward= float(sys.argv[2]) #specifies the reward of any state that is non-terminal.
    gamma = float(sys.argv[3])  #specifies the value of γ that you should use in the utility formulas.
    K = int(sys.argv[4]) #specifies the number of times that the main loop of the algorithm should be iterated

    if os.path.exists(environment_file):
        file_train= open(environment_file, "r")
        
        if os.path.isfile(environment_file):
            #reading from file and declaring array dimensions
            input_arr = np.genfromtxt(file_train, delimiter = ',', dtype= str);
            columns_train= len(input_arr[0])    #total number of columns in the file
            rows_train= len(input_arr)          #total number of rows in the file
            
            # INITIALIZING
            Actions = np.array(["<",">","v","^"]);
            Up= np.zeros(shape = [rows_train, columns_train], dtype= float)
            
            #====================== MAIN LOOP =====================================
            for z in range(K): #main loop iteration, K is hyperparameter
                U= copy.deepcopy(Up) #deepcopy so we don't rewrite the same address
                arr_actions_taken = []  #this will hold our direction arrows for where we should go in any state
                for i in range(rows_train): #looping through all states
                    for j in range(columns_train):
                        reward= REWARD(input_arr[i][j], non_terminal_reward)
                        
                        #if it is a non-terminal state
                        if( input_arr[i][j] == "." ):
                            #U'[s]= R(s) + 𝛾 * (max utility over all actions of [∑s'{ p(s'|s,a)*U[s'] } ])
                            maximum, argmax= MAX_EXPECTED_ALL_ACTIONS(Actions, U, i, j, rows_train, columns_train, input_arr)
                            Up[i][j]= reward + (gamma*(maximum)) #adding to new array: reward + Expected utility for a state over all possibles actions in that state
                            arr_actions_taken.append(Actions[argmax]) #appending the directional arrow
                        #if its an unreachable state, has an X
                        elif( input_arr[i][j] == "X" ):
                            arr_actions_taken.append("x")
                            Up[i][j]= 0
                            
                        #if its a terminal state
                        else: 
                            arr_actions_taken.append("o")
                            Up[i][j]= reward
                            
            AT= np.reshape(arr_actions_taken, (rows_train, columns_train)) #1d array into 2d
            
            print("\nutilities:\n")
            for i in range(rows_train):
                for j in range(columns_train):
                    print("%6.3f\t"%(Up[i][j]), end="")
                print("\n")
            
            print("\npolicy:", end="")
            for i in range(len(arr_actions_taken)):
                if( i %columns_train ==0):
                    print("\n")
                print("%s\t"%(arr_actions_taken[i]), end="")
                

            
        else:
            print(environment_file," is not a file.")
            exit(1)
            
    else:
        print("'", environment_file, "' Path does not exist")
        exit(1)
        
def REWARD(state, non_terminal_reward):
    if state == ".":
        return non_terminal_reward
    elif state == "X":
        return 0;
    else:
        try:
            float(state)
            return float(state)
        except ValueError:
            return 0

#logic for the second part of the equation: U'[s]= R(s) + 𝛾 * (max utility over all actions of [∑s'{ p(s'|s,a)*U[s'] } ])
#(for all actions(sum the state's weighted probabilities*new states utility for weighted probability of the chance you will move in the chosen direction or perpendicular to that direction.))

def MAX_EXPECTED_ALL_ACTIONS(Actions, U, row, column, total_rows, total_columns, input_arr):
    action_arr= np.empty([len(Actions)], dtype=float)
    for a in range(len(Actions)):
        if(Actions[a] == "<"):
            utility= 0
            if ((column-1)<0) or input_arr[row][column-1] == "X":
                utility+= 0.8*U[row][column]
            else:
                utility+= 0.8*U[row][column-1]
                
            if(  (( row+1 ) > ( total_rows-1 )) or input_arr[row+1][column] == "X"  ):       #probability of going down
                utility+= 0.1*U[row][column]
            else:
                utility+= 0.1*U[row+1][column]
                
            if( (( row-1 ) < 0) or input_arr[row-1][column] == "X" ):       #probability of going down
                utility+= 0.1*U[row][column]
            else:
                utility+= 0.1*U[row-1][column]
            action_arr[0]=utility
            
        elif(Actions[a] == ">"):
            utility=0
            if ( ((column+1) > (total_columns-1)) or input_arr[row][column+1] == "X"):
                utility+= 0.8*U[row][column]
            else:
                utility+= 0.8*U[row][column+1]
                
            if(  (( row+1 ) > ( total_rows-1 )) or input_arr[row+1][column] == "X" ):       #probability of going down
                utility+= 0.1*U[row][column]
            else:
                utility+= 0.1*U[row+1][column]
                
            if( (( row-1 ) < 0) or input_arr[row-1][column] == "X" ):       #probability of going down
                utility+= 0.1*U[row][column]
            else:
                utility+= 0.1*U[row-1][column]
            action_arr[1]=utility
            
        elif(Actions[a] == "v"):
            utility=0
            if (((row+1)> (total_rows-1)) or input_arr[row+1][column] == "X"):
                utility+= 0.8*U[row][column]
            else:
                utility+= 0.8*U[row+1][column]
                
            if((( column+1 ) > ( total_columns-1 )) or input_arr[row][column+1] == "X"):       #probability of going down
                utility+= 0.1*U[row][column]
            else:
                utility+= 0.1*U[row][column+1]
                
            if( (( column-1 ) < 0) or input_arr[row][column-1] == "X"):       #probability of going down
                utility+= 0.1*U[row][column]
            else:
                utility+= 0.1*U[row][column-1]
            action_arr[2]=utility
            
        elif(Actions[a] == "^"):
            utility=0
            if (((row-1)< ( 0 )) or input_arr[row-1][column] == "X"):
                utility+= 0.8*U[row][column]
            else:
                utility+= 0.8*U[row-1][column]
                
            if((( column+1 ) > ( total_columns-1 )) or  input_arr[row][column+1] == "X" ):       #probability of going down
                utility+= 0.1*U[row][column]
            else:
                utility+= 0.1*U[row][column+1]
                
            if( (( column-1 ) < 0) or input_arr[row][column-1] == "X" ):       #probability of going down
                utility+= 0.1*U[row][column]
            else:
                utility+= 0.1*U[row][column-1]
            action_arr[3]=utility

    return np.amax(action_arr), np.argmax(action_arr)
            
            
value_iteration()
    
