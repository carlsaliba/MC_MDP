from antlr4 import *
from gramLexer import gramLexer
from gramListener import gramListener
from gramParser import gramParser
import networkx as nx
import matplotlib.pyplot as plt
import random

import pydot


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


if __name__ == "__main__":
    print("Test file is running.")

    # Load MDP file
    mdp_file = "ex.mdp"
    states, actions, transitions = parse_mdp_file(mdp_file)

    # Print extracted information
    print("States:", states)
    print("Actions:", actions)
    print("Transitions:", transitions)

    G = nx.MultiDiGraph()

    # Add states as main nodes
    states_to_add = list(dict.fromkeys(state for state, _ in transitions.keys())) # SOlVED

    G.add_nodes_from(states_to_add)

    # Dictionary to store action nodes
    action_nodes = {}

    print(f'Nodes:\n{G.nodes()}\n')

    # Add edges: State → Action (deterministic) and Action → Next State (probabilistic)
    for (state, action), next_states in transitions.items():
        if action is not None:
            action_node = f"{state}_{action}"  # Create a unique action node (e.g., "S1_a")
            action_nodes[action_node] = action
            G.add_node(action_node, shape="square")  # Different shape for action nodes
            G.add_edge(state, action_node, weight=1.0, label=action)  # Deterministic transition

            for next_state, prob in next_states.items():
                if prob > 0.0:
                    G.add_edge(action_node, next_state, weight=prob, label=f"{prob:.2f}")

        else:  # Direct state-to-state transition (without action)
            for next_state, prob in next_states.items():
                if prob > 0.0:
                    G.add_edge(state, next_state, weight=prob, label=f"∅ ({prob:.2f})")


    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='neato')

    # Export to DOT file
    nx.drawing.nx_pydot.write_dot(G, 'visu/markov_arcs.dot')

    # Generate PNG using Pydot
    (graph,) = pydot.graph_from_dot_file('visu/markov_arcs.dot')
    graph.write_png('visu/markov_arcs.png')

    print("MDP graph saved as 'visu/markov_arcs.png'")