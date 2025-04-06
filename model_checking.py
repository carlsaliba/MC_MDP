import numpy as np
import random
from test import parse_mdp_file
from numpy.linalg import inv
from numpy.linalg import pinv

from numpy.linalg import solve
import networkx as nx
import pandas as pd

import math

def dict_to_matrix(states, transitions):
    lignes = [key for key,value in transitions.items()]
    matrice = np.zeros((len(lignes), len(states)))

    
    for key1,dict in transitions.items():
        for key2, proba in dict.items():
            
            i = lignes.index(key1)
            j = states.index(key2)
            matrice[i][j] = proba
    


    return matrice, lignes

def find_unreachable_states(G, target_states):
    reachable_from = set()
    for target in target_states:
        if target in G:
            reachable_from.update(nx.ancestors(G, target))
            reachable_from.add(target)  # The target itself is reachable
    return [node for node in G.nodes if node not in reachable_from]


# Function to compute the probability of reaching the target states in exactly 1 step
def compute_one_step_probabilities(G, target_states):
    one_step_probs = {node: 0.0 for node in G.nodes}  

    for node in G.nodes:
        if node in target_states:
            one_step_probs[node] = 1.0  # Target states have probability 1
            continue

        total_prob = sum(G[node][neighbor]['weight'] for neighbor in G.successors(node) if neighbor in target_states)
        one_step_probs[node] = total_prob

    return one_step_probs



def linear(proba_matrix,states,target_states):
    G = nx.DiGraph()

    G.add_nodes_from(states)

    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            if proba_matrix[i, j] > 0:
                G.add_edge(from_state, to_state, weight=proba_matrix[i, j])



    unreachable_states = find_unreachable_states(G, target_states)
    print("Unreachable states:", unreachable_states)

    G_simplified = G.copy()
    G_simplified.remove_nodes_from(unreachable_states)


    one_step_vector = compute_one_step_probabilities(G_simplified, target_states)

    df_one_step = pd.DataFrame.from_dict(one_step_vector, orient='index', columns=['Probability to reach target in 1 step'])
    print(df_one_step)

    S1_states={state for state, prob in one_step_vector.items() if prob == 1.0}
    print(S1_states)
    

    states_to_remove = set(unreachable_states) | S1_states

    remaining_states = [state for state in states if state not in states_to_remove]
    remaining_indices = [states.index(state) for state in remaining_states]

    s_int = proba_matrix[np.ix_(remaining_indices, remaining_indices)]

    one_step_vector = compute_one_step_probabilities(G_simplified, list(S1_states))
    vector_b = np.array([value for _, value in one_step_vector.items() if value != 1])


    print(s_int)
    print(vector_b)

    I=np.eye(s_int.shape[0], dtype=float)

    sol = np.linalg.solve(I - s_int, vector_b)
    print(sol)


def iter(proba_matrix,states,target_states,iteration):
    G = nx.DiGraph()

    G.add_nodes_from(states)

    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            if proba_matrix[i, j] > 0:
                G.add_edge(from_state, to_state, weight=proba_matrix[i, j])

    unreachable_states = find_unreachable_states(G, target_states)
    print("Unreachable states:", unreachable_states)

    G_simplified = G.copy()
    G_simplified.remove_nodes_from(unreachable_states)


    one_step_vector = compute_one_step_probabilities(G_simplified, target_states)

    df_one_step = pd.DataFrame.from_dict(one_step_vector, orient='index', columns=['Probability to reach target in 1 step'])
    print(df_one_step)

    S1_states={state for state, prob in one_step_vector.items() if prob == 1.0}
    print(S1_states)
    

    states_to_remove = set(unreachable_states) | S1_states

    remaining_states = [state for state in states if state not in states_to_remove]
    remaining_indices = [states.index(state) for state in remaining_states]

    s_int = proba_matrix[np.ix_(remaining_indices, remaining_indices)]

    one_step_vector = compute_one_step_probabilities(G_simplified, list(S1_states))
    vector_b = np.array([value for _, value in one_step_vector.items() if value != 1])


    print(s_int)
    print(vector_b)

    x_0=np.zeros(s_int.shape[0], dtype=float)

    x=np.dot(s_int,x_0)+vector_b

    iteration=iteration-1
    for i in range(iteration):

        x=np.dot(s_int,x)+vector_b



    print(x)


def choose_next_state(state, action):
    transition_probs = transitions.get((state, action))
    if transition_probs:
        next_state = random.choices(list(transition_probs.keys()), weights=transition_probs.values())[0]
        return next_state
    return state  # Stay in the same state if no transition found




def simulate_SMC(start_state='I',steps=5,delta=0.01,epsilon=0.01,state_actions=None,target_states=None):

    num_of_simu = math.ceil((math.log(2) - math.log(delta)) / ((2 * epsilon) ** 2))
    Y_n=0
    for j in range(num_of_simu):
        state = start_state
        for i in range(steps):
            #print(f"Step {i}: Current State → {state}")

            available_actions = state_actions[state]
            action = None
            if available_actions:
                action = random.choice(available_actions)
            
            next_state = choose_next_state(state, action)


            state = next_state
        if state in target_states:
            Y_n=Y_n+1

    gamma_N=Y_n/num_of_simu
    
    lower_bound = gamma_N - epsilon
    upper_bound = gamma_N + epsilon
    print(f"gamma n is equal to :{Y_n}")
    return lower_bound, upper_bound

def sprt(start_state='I', steps=10, theta0=0.5, theta1=0.8, alpha=0.05, beta=0.05, state_actions=None, target_states=None):
    
    # Constants for the test
    A = (1 - beta) / alpha
    B = beta / (1 - alpha)
    
    # Running statistics
    ratio = 1.0
    successes = 0
    samples = 0
    
    while True:

        samples += 1
        state = start_state
        reached_target = False
        
        
        for _ in range(steps):
            available_actions = state_actions[state]
            action=None
            if available_actions:
                action = random.choice(available_actions)

            
            next_state = choose_next_state(state, action)
            state = next_state
            
            if state in target_states:
                reached_target = True
                break
        
        if reached_target:
            successes += 1
            ratio *= theta1 / theta0
        else:
            ratio *= (1 - theta1) / (1 - theta0)
        
        # Check stopping condition
        if ratio >= A:
            return 1, successes / samples, samples  # Accept H1
        elif ratio <= B:
            return 0, successes / samples, samples  # Accept H0
        
        if samples >= 10000:  # Arbitrary large number
            probability_estimate = successes / samples
            return None, probability_estimate, samples





if __name__ == "__main__":
    print("Test file is running.")

    mdp_file = "dé_par_monnaies.mdp"
    states, actions, transitions = parse_mdp_file(mdp_file)

    print("States:", states)
    print("Actions:", actions)
    print("Transitions:", transitions)



    statess = set(state for state, _ in transitions.keys())

    state_actions = {state: set() for state in statess}

    for (state, action) in transitions:
        if action is not None:
            state_actions[state].add(action)

    # Convert sets to lists and handle states with no actions
    state_actions = {state: list(actions) if actions else None for state, actions in state_actions.items()}



    matrice, lignes = dict_to_matrix(states, transitions)
    for (etats, act) in lignes:
        if etats not in states or act not in actions and act != None:

            raise ValueError(f"Erreur : L'état '{etats}' ou l'action '{act}' n'est pas défini.")
    
    
    #je transforme les tuples en liste simple puisque dans une MP on a pas des actions
    lignes = [key[0] for key,value in transitions.items()]

    print("lignes:\n", lignes)
    print("matrice:\n", matrice)


    # Ask the user to input target states as comma-separated values
    user_input = input("Enter target states separated by commas (e.g., W,X,Y): ")

    # Convert the user input into a set
    target_states = set(user_input.split(','))

    print("Target States:", target_states)
    states = lignes
    proba_matrix =matrice

    method_choice = input("Choose a method (simulate, iter, linear, SPRT): ").strip().lower()

    if method_choice == "simulate":
        start_state = input("Enter the start state (e.g., S0): ").strip()
        steps = int(input("Enter the number of steps: "))
        delta = float(input("Enter delta value (e.g., 0.01): "))
        epsilon = float(input("Enter epsilon value (e.g., 0.01): "))

        result = simulate_SMC(start_state, steps, delta, epsilon, state_actions, target_states)
        print("Lower Bound:", result[0], "Upper Bound:", result[1])

    elif method_choice == "iter":
        iterations = int(input("Enter the number of iterations: "))
        iter(matrice, lignes, target_states, iterations)

    elif method_choice == "linear":
        linear(matrice, lignes, target_states)

    elif method_choice == "sprt":
        start_state = input("Enter the start state (e.g., S0): ").strip()
        steps = int(input("Enter the number of steps: "))
        proba=float(input("Enter the proba: "))
        eps=0.05
        sprt_decision, sprt_prob, sprt_samples = sprt(
        start_state, steps, theta0=proba+eps, theta1=proba-eps, alpha=0.01, beta=0.01,state_actions=state_actions, target_states=target_states)
        decision_text = "Accept H1 (p < theta1)" if sprt_decision == 1 else \
                   "Accept H0 (p > theta0)" if sprt_decision == 0 else \
                   "Inconclusive"
    
        print(f"SPRT: Decision: {decision_text}, probability estimate: {sprt_prob:.4f}, used {sprt_samples} samples")

    else:
        print("Invalid choice. Please select 'simulate', 'iter', or 'linear'.")
