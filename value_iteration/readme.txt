Value Iteration algorithm is a reinforcement algorithm that finds the best path in an enviroment to a terminal state.

Implement the value iteration algorithm.
Arguments
value_iteration(<environment_file>, <non_terminal_reward>, <gamma>, <K>) 
The first argument, <environment_file>, is the path name of a file that describes the environment where the agent moves (see details below). The path name can specify any file stored on the local computer.
The second argument, <non_terminal_reward>, specifies the reward of any state that is non-terminal.
The third argument, <gamma>, specifies the value of γ that you should use in the utility formulas.
The fourth argument, <K>, specifies the number of times that the main loop of the algorithm should be iterated. The initialization stage, where U[s] is set to 0 for every state s, does not count as an iteration. After the first iteration, if you implement the algorithm correctly, it should be the case that U[s]=R[s].

Figure 1: The environment described in file environment1.txt.
The environment file will follow the same format as files environment1.txt and environment2.txt. For example, file environment1.txt describes the world shown in Figure 1, and it has the following contents:

1.0,X
.,-1.0

Figure 2: The environment described in file environment2.txt.
Similarly, file environment2.txt describes the world shown in Figure 2, and it has the following contents:

.,.,.,1.0
.,X,.,-1.0
.,.,.,.
As you see from the two examples, the environment files are CSV (comma-separated values) files, where:

Character '.' represents a non-terminal state.
Character 'X' represents a blocked state, that cannot be reached from any other state. You can assume that blocked states have utlity value 0.
Numbers represent the rewards of TERMINAL states. So, if the file contains a number at some position, it means that that position is a terminal state, and the number is the reward for reaching that state. 
These rewards are real numbers, they can have any value.
