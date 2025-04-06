import networkx as nx
import pydot
import numpy as np
from test import parse_mdp_file
import random
import time

def value_iteration(states, transitions, rewards, gamma=0.9, theta=1e-6):
    V = {state: 0 for state in states}  # Initialize value function
    policy = {state: None for state in states}  # Store best action

    state_actions = {state: set() for state in states}

    for (state, action) in transitions:
        if action is not None:
            state_actions[state].add(action)

    state_actions = {state: list(actions) if actions else None for state, actions in state_actions.items()}
    
    print(state_actions)

    for _ in range(1000):
        delta = 0  # Track convergence
        for state in states:
            
            if state not in rewards:
                continue

            max_value = float('-inf')
            best_action = None

            available_actions = state_actions[state]
            if available_actions !=None :
                
                for action in available_actions:
                    if (state, action) in transitions:
                        
                        expected_value = rewards[state] + gamma*sum(prob * (V[next_state] )
                                            for next_state, prob in transitions[(state, action)].items())
                        if expected_value > max_value:
                            max_value = expected_value
                            best_action = action

                delta = max(delta, abs(max_value - V[state]))
                V[state] = max_value
                policy[state] = best_action

            else:
                expected_value = rewards[state] + gamma*sum(prob * (V[next_state] )
                                        for next_state, prob in transitions[(state, None)].items())
                if expected_value > max_value:
                    max_value = expected_value
                    best_action = action

                V[state] = max_value

        
    return V, policy



def choose_next_state(state, action):
    transition_probs = transitions.get((state, action))
    random.seed(None)  # Uses OS entropy
    if transition_probs:
        next_state = random.choices(list(transition_probs.keys()), weights=transition_probs.values())[0]
        return next_state
    return state  # Stay in the same state if no transition found

def q_learning(states, transitions, rewards, episodes=100, alpha=0.1, gamma=1, epsilon=0.1):
    
    
    Q = {(state, action): 0 for state, action in transitions.keys()}  # Initialize Q-values

    policy = {state: None for state in states}
    
    state_actions = {state: set() for state in states}

    for (state, action) in transitions:
        if action is not None:
            state_actions[state].add(action)

    state_actions = {state: list(actions) if actions else None for state, actions in state_actions.items()}    
    print(state_actions)
    for _ in range(episodes):


        current_state = np.random.choice(list(states))
        
        available_actions = state_actions[current_state]

        action = None
        if available_actions:
            action = random.choice(available_actions)

        next_state = choose_next_state(current_state, action)
        reward = rewards.get(current_state, 0)
        
        next_actions = state_actions[next_state]
        
    
        next_max_q = max([Q.get((next_state, a), 0) for a in next_actions]) if next_actions else Q.get((next_state, None))
        
        # Update Q-value
        Q[(current_state, action)] += alpha * (reward + gamma * next_max_q - Q[(current_state, action)])
        
            
    
    # Determine optimal policy based on final Q-values
    for state in states:
        actions = state_actions.get(state, [])
        if actions:
            policy[state] = max(actions, key=lambda a: Q.get((state, a), 0))
    
    return Q, policy

    






if __name__ == "__main__":
    print("Test file is running.")
    
    # Load MDP file
    mdp_file = "ex2.mdp"
    states, actions, transitions = parse_mdp_file(mdp_file)

     # Print extracted information
    print("States:", states)
    print("Actions:", actions)
    print("Transitions:", transitions)

    # Define rewards manually
    rewards = {
        "S0":0,
        "S1":5,
        "S2":100,
        "S3" :500,
        "S4" :3
        
    }
    
    # Run Value Iteration
    optimal_values, optimal_policy = value_iteration(states, transitions, rewards)
    
    print("Optimal Values:", optimal_values)
    print("Optimal Policy:", optimal_policy)

        # Run Q-learning
    #q_values, q_policy = q_learning(states, transitions, rewards)
    #print("Q-values:", q_values)
    #print("Q-learning Policy:", q_policy)





