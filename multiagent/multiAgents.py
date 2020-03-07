# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ## Q1 
        newFoodList = newFood.asList()
        MinFoodDistance = -1
        for food in newFoodList:
            distance = util.manhattanDistance(newPos, food)
            if MinFoodDistance >= distance or MinFoodDistance == -1:
                MinFoodDistance = distance
        ## Distance from PacMan to ghosts is calculated 
        Dist_PacMan_Ghost = 1
        ## Checking if the ghost is near (~ distance of 1) to PacMan
        Ghost_Around_Pac = 0
    
        for ghost_state in successorGameState.getGhostPositions():
            distance = util.manhattanDistance(newPos, ghost_state)
            Dist_PacMan_Ghost += distance
            if distance <= 1:
                Ghost_Around_Pac += 1
        ## Returning the calculated score 
        return successorGameState.getScore() + (1 / float(MinFoodDistance)) - (1 / float(Dist_PacMan_Ghost)) - Ghost_Around_Pac
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        ## defined few more functions to get the result 
        return self.minimax_value(gameState,0,0)
        
    def minimax_value(self, gameState, agentIndex, nodeDepth):
    ## this function checks the depth and agent index and based on that we decide which level it is in. Based on that we send it on the min/max value function. Whether we need to check the min value in that level or max value in that level 

        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            nodeDepth += 1

        if nodeDepth == self.depth:
            return self.evaluationFunction(gameState)

        if agentIndex == self.index:
            return self.max_value(gameState, agentIndex, nodeDepth)
        else:
            return self.min_value(gameState, agentIndex, nodeDepth)

        return 'None'
     
    def max_value(self, gameState, agentIndex, nodeDepth):
        ## If its on the level where we need to check the max value. We run this function where we update the value and check for max value for the parent node
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        value = float("-inf")
        actionValue = "None"
        legalActions = gameState.getLegalActions(agentIndex)
        
        for actions in legalActions:
            if actions == Directions.STOP:
                continue;
            successor = gameState.generateSuccessor(agentIndex,actions)
    #     if nodeDepth==self.depth:
    #       temp = self.evaluationFunction(gameState)
            temp = self.minimax_value(successor,agentIndex+1,nodeDepth)

            if temp > value:
                value = max(temp,value)
                actionValue = actions

        if nodeDepth == 0:
            return actionValue
        else:
            return value

    def min_value(self, gameState, agentIndex, nodeDepth):
        ## If its on the level where we need to check the min value. We run this function where we update the value and check for min value for the parent node and then send it back to minmax function
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        value = float("inf")
        actionValue = "None"

        legalActions = gameState.getLegalActions(agentIndex)
        agentNumber = gameState.getNumAgents()

        for actions in legalActions:
            if actions == Directions.STOP:
                continue;
            successor = gameState.generateSuccessor(agentIndex,actions)
    #     if agentIndex == agentNumber-1:

    #       if nodeDepth==self.depth:
    #         temp = self.evaluationFunction(successor)

    #       else:
    #         temp = self.max_value(successor,0,nodeDepth+1)
    #     else:
    #       temp = self.min_value(successor,agentIndex+1,nodeDepth)
            temp = self.minimax_value(successor,agentIndex+1,nodeDepth)

            if temp < value:
                value = min(temp,value)
                actionValue = actions
        return value
    #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maximizer(agent, depth, game_state, a, b):  
            # maximizer function
            v = float("-inf")
            for newState in game_state.getLegalActions(agent):
                v = max(v, alphabetaprune(1, depth, game_state.generateSuccessor(agent, newState), a, b))
                if v > b:
                    return v
                a = max(a, v)
            return v

        def minimizer(agent, depth, game_state, a, b):  
            # minimizer function
            v = float("inf")

            next_agent = agent + 1  
            # calculate the next agent and increase depth accordingly.
            if game_state.getNumAgents() == next_agent:
                next_agent = 0
            if next_agent == 0:
                depth += 1

            for newState in game_state.getLegalActions(agent):
                v = min(v, alphabetaprune(next_agent, depth, game_state.generateSuccessor(agent, newState), a, b))
                if v < a:
                    return v
                b = min(b, v)
            return v

        def alphabetaprune(agent, depth, game_state, a, b):
            if game_state.isLose() or game_state.isWin() or depth == self.depth:  
                # return the utility in case the defined depth is reached or the game is won/lost.
                return self.evaluationFunction(game_state)

            if agent == 0: 
                # maximize for pacman
                return maximizer(agent, depth, game_state, a, b)
            else:  
                # minimize for ghosts
                return minimizer(agent, depth, game_state, a, b)

   ## Performing maximizer function to the root node i.e. pacman using alpha-beta pruning
        utility = float("-inf")
        action = Directions.WEST
        alpha = float("-inf")
        beta = float("inf")
        for agentState in gameState.getLegalActions(0):
            ghostValue = alphabetaprune(1, 0, gameState.generateSuccessor(0, agentState), alpha, beta)
            if ghostValue > utility:
                utility = ghostValue
                action = agentState
            if utility > beta:
                return utility
            alpha = max(alpha, utility)

        return action

     ##   util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgent = gameState.getNumAgents()
        ActionScore = []

        def RMStop(List):
            return [x for x in List if x != 'Stop']

        def _expectMinimax(s, iterCount):
            if iterCount >= self.depth*numAgent or s.isWin() or s.isLose():
                ## return the heuristic value of node
                return self.evaluationFunction(s)
            #Ghost min
            if iterCount%numAgent != 0: 
                successorScore = []
                for a in RMStop(s.getLegalActions(iterCount%numAgent)):
                    sdot = s.generateSuccessor(iterCount%numAgent,a)
                    result = _expectMinimax(sdot, iterCount+1)
                    successorScore.append(result)
                    averageScore = sum([ float(x)/len(successorScore) for x in successorScore])
                return averageScore
            # Pacman Max
            else: 
                result = -1e10
                for a in RMStop(s.getLegalActions(iterCount%numAgent)):
                    sdot = s.generateSuccessor(iterCount%numAgent,a)
                    result = max(result, _expectMinimax(sdot, iterCount+1))
                    if iterCount == 0:
                        ActionScore.append(result)
                return result
          
        result = _expectMinimax(gameState, 0);
        return RMStop(gameState.getLegalActions(0))[ActionScore.index(max(ActionScore))]

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Calculating the distances from pacman to the ghosts, also check for thedistance of the ghosts (at distance of 1) around pacman.
    Giving higher weightage to the state with less number of food 
    Giving higher risk state i.e. where the pacmac gets caught up with a ghost a lower weightage
    GEt the min distance between the pacman and a ghost and also check the number of capsules in a state
    """
    "*** YOUR CODE HERE ***"
    curPos = currentGameState.getPacmanPosition()
    curFoodList = currentGameState.getFood().asList()
    curFoodCount = currentGameState.getNumFood()
    curGhostStates = currentGameState.getGhostStates()
    curScaredTimes = [ghostState.scaredTimer for ghostState in curGhostStates]
    curCapsules = currentGameState.getCapsules()
    curScore = currentGameState.getScore()

    foodLeft = 1.0/(curFoodCount + 1.0)
    ghostDist = float("inf")
    scaredGhosts = 0

    # print curScaredTimesS

    for ghostState in curGhostStates:
        ghostPos = ghostState.getPosition()
        if curPos == ghostPos:
            return float("-inf")
        else:
            ghostDist = min(ghostDist,manhattanDistance(curPos,ghostPos))
      
        if ghostState.scaredTimer != 0:
            scaredGhosts += 1

    capDist = float("inf")
    for capsuleState in curCapsules:
        capDist = min(capDist,manhattanDistance(curPos,capsuleState))

    ghostDist = 1.0/(1.0 + (ghostDist/(len(curGhostStates))))
    # capDist = 1.0/(1.0 + capDist)
    capDist = 1.0/(1.0 + len(curCapsules))
    scaredGhosts = 1.0/(1.0 + scaredGhosts)


    return curScore + (foodLeft + ghostDist + capDist)
##    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
