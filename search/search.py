# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import Stack
from util import Queue,PriorityQueue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    ## Get the start state 
    start = problem.getStartState()
    ## state keeps a check on the states we want to expand
    ## For the first position initialize everything 
    #print(1)
    state = Stack() 
    state.push(start)
    ## visited stack keeps a check of all the states/nodes we are visiting
    #print(1)
    Visited = []
    ## Path keeps the list of the path or directions we are following to reach the goal state 
    #print(1)
    Route=[]
    actions=Stack()
    #print(1)
    position = state.pop()
    ## run a loop till we reach the goal state
    #print(1)
    while not problem.isGoalState(position):
        if position not in Visited:
            #print(1)
            Visited.append(position) ## update the variables
            successors = problem.getSuccessors(position) ## Got to the successor
            for successor_node,direction,cost in successors:
                #print(2)
                state.push(successor_node)
                tempRoute = Route + [direction]
                actions.push(tempRoute)
        #print(1)
        position = state.pop()
        Route = actions.pop()
    return Route
    util.raiseNotDefined()

    
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    ## state keeps a check on the states we want to expand
    ## For the first position initialize everything 
    #print(1)
    # state (Queue) to store the nodes along with their paths
    state = util.Queue()   
    # Pushing (Node, [Path from start-node till 'Node']) to the state
    state.push((start, []))
    ## visited keeps a check of all the states/nodes we are visiting
    # A set to maintain all the visited nodes
    Visited = set()
    ## Path keeps the list of the path or directions we are following to reach the goal state 
    ## run a loop till we reach the goal state
    while True:
        last_element = state.pop()
        node = last_element[0]
        actions = last_element[1]
        # Exit on encountering goal node
        if problem.isGoalState(node):   
            break
        else:
            # Skipping already visited nodes
            if node not in Visited:  
                # Adding newly encountered nodes to the set of visited nodes
                Visited.add(node)    
                successors = problem.getSuccessors(node)
                for successor in successors:
                    child_node = successor[0]
                    child_path = successor[1]
                    # Computing path of child node from start node
                    route = actions + [child_path] 
                    # Pushing ('Child Node',[Full Path]) to the fringe
                    state.push((child_node, route))   
    return actions

    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start = problem.getStartState()
    # State to manage which states to expand
    state = PriorityQueue()                    
    state.push(start,0)
    # List to check whether state has already been visited
    Visited = []                                
    # Temp variable to get intermediate paths
    actions=[]                            
    # List to store final sequence of directions    
    path=[]                    
    # Queue to store direction to children (currState and Route go hand in hand)
    Route=PriorityQueue()               
    currState = state.pop()
    while not problem.isGoalState(currState):
        if currState not in Visited:
            Visited.append(currState)
            successors = problem.getSuccessors(currState)
            #Check for all the successor nodes 
            for child,direction,cost in successors:
                actions = path + [direction]
                costToGo = problem.getCostOfActions(actions)
                if child not in Visited:
                    state.push(child,costToGo)
                    Route.push(actions,costToGo)
        currState = state.pop()
        path = Route.pop() 
    #return the final path
    return path
    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    ## initializeing the variables
    start = problem.getStartState()
    Visited = []
    ## Using Priority Queue and heuristic
    state = util.PriorityQueue()
    state.push((start, []), nullHeuristic(start, problem))
    nCost = 0
    while not state.isEmpty():
        position, actions = state.pop()
        if problem.isGoalState(position):
            ## Run the loop till we reach the goal
            return actions
        if position not in Visited:
            successors = problem.getSuccessors(position)
            for successor in successors:
                coordinates = successor[0]
                ## check if already visited the node or its new 
                if coordinates not in Visited:
                    directions = successor[1]
                    nActions = actions + [directions]
                    nCost = problem.getCostOfActions(nActions) + heuristic(coordinates, problem)
                    state.push((coordinates, actions + [directions]), nCost)
        Visited.append(position)
    return actions
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
