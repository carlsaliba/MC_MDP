from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import pydot
from scipy.optimize import linprog
from math import log
import mdptoolbox
import cvxpy as cp

class MarkovChainParser(gramListener):
    def __init__(self):
        self.states = []
        self.actions = []
        self.transitions = {}

    def enterDefstates(self, ctx):
        self.states = [str(x) for x in ctx.ID()]

    def enterDefactions(self, ctx):
        self.actions = [str(x) for x in ctx.ID()]

    def enterTransact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        act = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        
        # Normalize weights to probabilities
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        # Store transition (state, action) -> {next_state: probability}
        if (dep, act) not in self.transitions:
            self.transitions[(dep, act)] = {}
        for next_state, prob in zip(ids, probs):
            self.transitions[(dep, act)][next_state] = prob

    def enterTransnoact(self, ctx):
        ids = [str(x) for x in ctx.ID()]
        dep = ids.pop(0)
        weights = [int(str(x)) for x in ctx.INT()]
        
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]

        # Store transition without an action
        if (dep, None) not in self.transitions:
            self.transitions[(dep, None)] = {}
        for next_state, prob in zip(ids, probs):
            self.transitions[(dep, None)][next_state] = prob


def parse_mdp_file(file_path):
    input_stream = FileStream(file_path)
    lexer = gramLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = gramParser(stream)
    tree = parser.program()

    mc_parser = MarkovChainParser()
    walker = ParseTreeWalker()
    walker.walk(mc_parser, tree)

    return mc_parser.states, mc_parser.actions, mc_parser.transitions

def dict_to_matrix(states, transitions):
    lignes = [key for key,value in transitions.items()]
    matrice = np.zeros((len(lignes), len(states)))

    
    for key1,dict in transitions.items():
        for key2, proba in dict.items():
            i = lignes.index(key1)
            j = states.index(key2)
            matrice[i][j] = proba
    
    return matrice, lignes


def simulate_markov_chain(initial_state, matrice, lignes,states, num_steps=15):
    current_state = initial_state
    states_visited = [current_state]
    
    for _ in range(num_steps):
        # Sélectionner les transitions possibles pour l'état actuel
        actions_possible = [action for (etat ,action) in lignes if etat == current_state]
        if len(actions_possible)==0:
            current_action = None
        else:
            current_action = random.choice(actions_possible)
        if current_action is not None:
            i = lignes.index((current_state, current_action))
        else:
            i = lignes.index((current_state, None))
        possible_transitions = matrice[i]
        
        # Choisir un état suivant avec une probabilité selon les transitions
        next_state = random.choices(
            states,  # Les états cibles
            weights=possible_transitions,  # Les probabilités
            k=1  # On choisit un seul état
        )[0]
        
        # Ajouter l'état suivant à la liste des états visités
        states_visited.append(next_state)
        
        # Mettre à jour l'état actuel
        current_state = next_state
        actions_possible = []
    
    return states_visited


def calcul_proba_chain_de_markov(matrice, lignes, targets):
    # target correspond à l'indice des etats que l'on cherche à atteindre
    A = matrice
    # calcul de b
    b = np.zeros(matrice.shape[0])
    for i in range(matrice.shape[0]):
        x = 0
        for j in range(len(targets)):
            x += matrice[i,targets[j]]
        b[i] = x
    

    # réduire A
    S_0 = targets
    S_question = can_reach(matrice, targets)
    for i in range(A.shape[0]):
        if i not in S_question:
            S_0.append(i)
    matrice_modifiee = np.array(matrice)

    # Supprimer les lignes et les colonnes correspondant aux indices dans 'targets'
    matrice_modifiee = np.delete(matrice_modifiee, S_0, axis=0)  # Supprimer les lignes
    A = np.delete(matrice_modifiee, S_0, axis=1)
    b = np.delete(b, S_0, axis=0)
    print("A:\n",A)  
    # Résolution de Ay + b = y 
    print("b:\n",b)
    n = matrice_modifiee.shape[0]
    # Calcul de la matrice (A - I)
    I = np.eye(n)  # Matrice identité de taille n
    A_minus_I = I-A
    
    # Résolution du système 
    y = np.linalg.inv(A_minus_I)

    proba = np.linalg.vecdot(y,b)
    
    return proba, S_question

def can_reach(matrice, targets):
    pile = []
    S_question = []
    #target = indice des etats target
    for element in targets:
        pile.append(element)

    while len(pile) >0:
        j = pile.pop()
        for i in range(matrice.shape[0]):
            if matrice[i][j] > 0 and i!=j and i not in S_question:
                pile.append(i)
                S_question.append(i)
    S_question.sort()
    return S_question


def calcul_proba_MDP(matrice, lignes, targets, states, actions):
    """
    Cette fonction permet de calculer les proba optimal d'arriver dans target depuis chaque états qui peuvent (pas forcement) y arriver 
    """
    # créer la matrice A
    p = len(states)
    q = len(actions)
    A = np.zeros((p*(q+2), p))
    b = np.zeros(p*(q+2))
    for i in range(len(states)):
        A[i*(q+2)+q][i] = -1
        A[i*(q+2)+q+1][i] = 1
        b[i*(q+2)+q] = -1
        for j in range(len(actions)):

            if  (states[i], None) in lignes:
                 raise ValueError(f"Il n'y a pas d'action associé à l'états'{states[i]}'.")
            elif (states[i], actions[j]) in lignes:
                index = lignes.index((states[i], actions[j]))
                
                for k in range(p):
                
                    if i == k:
                        A[i*(q+2)+j][k] = 1-matrice[index][k]
                    else:
                        A[i*(q+2)+j][k] = -matrice[index][k]

    #print("Matrice MDP:\n", A)
    for i in range(len(lignes)):
        (etat,act) = lignes[i]
        indice_x = states.index(etat)
        indice_y = actions.index(act)
        for k in targets:
            b[indice_x*(q+2) + indice_y] += matrice[i][k] 
    #print("vecteur b:\n", b)
    
    block_a_enlever = []
    indice_a_enlever = []
    for i in range(p):
        proba_win = 0
        for j in range(q):
            for target in targets:
                proba_win += A[i*(q+2)+j][target]
        
        if proba_win == 0 or proba_win == q:
            indice_a_enlever.append(i)
            for j in range(q):
                block_a_enlever.append(i*(q+2)+j)
    last_four = [-1,-2,-3,-4]
    #remove the block inutile 
    new_A = np.delete(A, block_a_enlever, axis=0)
    new_A = np.delete(new_A, indice_a_enlever, axis=1)
    new_A = np.delete(new_A, last_four, axis=0)
    new_b = np.delete(b, block_a_enlever)
    new_b = np.delete(new_b, last_four)
    print("Matrice MDP:\n", new_A)
    print("vecteur b:\n", new_b)
    # Fonction objectif (minimiser 0, car on ne cherche pas à optimiser)

    p = new_A.shape[1]
    A = new_A
    b = new_b

    # Define the problem
    x = cp.Variable(p)

    # Define the constraints and objective
    constraints = [A @ x >= b]
    objective = cp.Minimize(cp.norm(x, 1))

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    #print("Optimal value:", problem.value)
    print("Optimal solution:", x.value)
    return A,b

def calcul_mdp(states, actions, transitions):
    n_states = len(states)
    n_actions = len(actions)

    # Initialisation des tableaux de probabilités et de récompenses
    P = np.zeros((n_actions, n_states, n_states))
    R = np.zeros((n_actions, n_states, n_states))

    for (state, action), outcomes in transitions.items():
        action_index = actions.index(action)
        state_index = states.index(state)
        for outcome_state, prob in outcomes.items():
            outcome_index = states.index(outcome_state)
            P[action_index, state_index, outcome_index] = prob
            # Ajoutez ici les valeurs de récompenses appropriées
            #R[action_index, state_index, outcome_index] = 0 if outcome_state == 'W' or outcome_state == 'L' else 1
            R[action_index, state_index, outcome_index] = 1 if outcome_state == 'W'  else 0

    discount_factor = 0.9
    
    # Création de l'objet PolicyIteration
    pi = mdptoolbox.mdp.PolicyIteration(P, R, discount_factor)

    # Exécution de l'itération de politique
    pi.run()

    # Affichage des résultats
    print("Adversaire optimale:\n",[actions[a] for a in pi.policy])
    return [actions[a] for a in pi.policy],P
    #print("Valeurs des états:")
    #print(pi.V)

def mdp_to_chaine_markov(policy_indices, P, actions):
    # Initialisation de la matrice de transition résultante
    P_combined = np.zeros_like(P[0])

    # Construire la matrice de transition combinée en utilisant la politique optimale
    for state in range(len(policy_indices)):
        action = policy_indices[state]
        idx = actions.index(action)
        P_combined[state] = P[idx][state]

    
    return P_combined



def smc_qualitatif():
    return 0


def smc_quantitatif( initial_state, matrice, lignes, states ,targets, delta = 0.01, eps = 0.01):
    N = int((log(2) - log(delta)) / 4/eps/eps) +1
    gamma_n = 0
    result = False
    for k in range(N):
        visited_states = simulate_markov_chain(initial_state, matrice, lignes,states, num_steps=15)
        for eta in targets:
            if eta in visited_states:
                result = True
        
        if result:
            gamma_n += 1
            result = False
    return gamma_n/N

def mdp_iteration(A, b, n_iter):
    # Ensure dimensions are compatible
    assert A.shape[0] == b.shape[0], "The number of rows in A must match the size of b"
    n, m = A.shape

    x = np.zeros(m)  # Initialize x with zeros, where x has m elements

    for k in range(n_iter):
        x = A @ x + b  # Use @ for matrix multiplication

    return x


if __name__ == "__main__":
    print("Test file is running.")

    # Load MDP file
    mdp_file = "ex10.mdp"
    states, actions, transitions = parse_mdp_file(mdp_file)
    
    # Print extracted information
    print("States:", states)
    print("Actions:", actions)
    print("Transitions:", transitions)

    matrice, lignes = dict_to_matrix(states, transitions)
    for (etats, act) in lignes:
        if etats not in states or act not in actions and act != None:
            raise ValueError(f"Erreur : L'état '{etats}' ou l'action '{act}' n'est pas défini.")

    print("lignes:\n", lignes)
    #print("matrice:\n", matrice)
    
    targets = ['W']
    
    #proba = smc_quantitatif( 'S0', matrice, lignes, states ,targets, delta = 0.01, eps = 0.01)
    #print("proba:")
    #print(proba)
    """y, S_question = calcul_proba_chain_de_markov(matrice, lignes, targets)
    print("S?:\n",S_question)
    print("proba:\n",y)"""
    #targets = ['W']
    """filtered_states = filter_states(states, actions, transitions, targets, matrice, lignes)
    print("filtered_states\n", filtered_states)"""
    targets = [3]
    A,b = calcul_proba_MDP(matrice, lignes, targets, states, actions)
    """x = mdp_iteration(A,b, 10)
    print("iter:\n", x)
    """
    adversaire,P = calcul_mdp(states, actions, transitions)
    """
    P_combined = mdp_to_chaine_markov(adversaire, P, actions)
    print("Matrice de transition résultante :\n", P_combined)
    y, S_question = calcul_proba_chain_de_markov(P_combined, lignes, targets)
    print("S?:\n",S_question)
    print("proba:\n",y)
    """