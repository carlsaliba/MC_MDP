import random
from test import parse_mdp_file
import imageio.v2 as imageio
import networkx as nx
import pydot
import random
import imageio.v2 as imageio
import matplotlib.pyplot as plt

def choose_next_state(state, action):
    transition_probs = transitions.get((state, action))
    if transition_probs:
        next_state = random.choices(list(transition_probs.keys()), weights=transition_probs.values())[0]
        return next_state
    return state  # Stay in the same state if no transition found

# Simulation function
def simulate(start_state='I', steps=10, save_path="images/simulation.gif"):
    state = start_state
    frames = []  # To store images

    for i in range(steps):
        print(f"Step {i}: Current State → {state}")

        # Choose an action (if available)
        available_actions = state_actions[state]
        action = None
        if available_actions:
            action = random.choice(available_actions)
           

        # Move to the next state
        next_state = choose_next_state(state, action)

        # Generate and save the frame
        frame_path = f"images/frame_{i}.png"
        draw_graph(G, state, frame_path)
        frames.append(imageio.imread(frame_path))

        if available_actions:
            frame_path = f"images/sub_frame_{i}.png"
            sub_state=f"{state}_{action}" 
            draw_graph(G, sub_state, frame_path)
            frames.append(imageio.imread(frame_path))


        # Update state
        state = next_state

    # Create a GIF from frames
    imageio.mimsave(save_path, frames, duration=1000)
    print(f"Animation saved as {save_path}")
    


def draw_graph(G, current_state, filename):


    # Set node attributes for coloring
    if current_state not in G.nodes:
        raise ValueError(f"Node '{current_state}' not found in the graph.")
    

    for node in G.nodes:
        G.nodes[node]["fillcolor"] = "white"  # Default fill color
        G.nodes[node]["color"] = "black"  # Ensure border is visible
        G.nodes[node]["style"] = "filled"  

    G.nodes[current_state]['fillcolor'] = "red"  # Highlight current state
    

    # Export to DOT file
    nx.drawing.nx_pydot.write_dot(G, filename)

    # Generate PNG using Pydot
    (graph,) = pydot.graph_from_dot_file(filename)

    # Apply color attributes from NetworkX graph to Pydot graph
    for node in graph.get_nodes():
        node_name = node.get_name().strip('"')
        if node_name in G.nodes:
            node.set_fillcolor(G.nodes[node_name]["fillcolor"])
            node.set_style("filled")
            node.set_color(G.nodes[node_name]["color"]) 

    # Save as PNG
    graph.write_png(filename)

    #print(f"MDP graph saved as {filename}")


def build_graph(transitions):

    G = nx.MultiDiGraph()

    # Add states as main nodes
    states_to_add = list(dict.fromkeys(state for state, _ in transitions.keys())) # SOlVED
    #sorted_states = sorted(states, key=lambda s: int(s[1:]))  # Sort by numeric part
    ## add a manner to tread S_0 and W or L for sorting and putting the last states at the end.
    G.add_nodes_from(states_to_add)

    # Dictionary to store action nodes
    action_nodes = {}

    print(f'Nodes:\n{G.nodes()}\n')

    for (state, action), next_states in transitions.items():
        if action is not None:
            action_node = f"{state}_{action}"  
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
    
    return G

def play_player(start_state='S0'):
        state = start_state
        while True:
            print(f"Current State: {state}")
            
            # If the state has actions, choose an action
            available_actions = state_actions[state]
            action = None
            if available_actions:
                print(f'Choose what action you want to play {available_actions}')
                action=input()
                
                #action = random.choice(available_actions)  # Choose an action randomly
            
            state = choose_next_state(state, action)
            
            print(f"  Chose Action: {action if action else 'None'} → Moving to: {state}")
            stop = input("Choose if you want to continue or stop.\nPress any to continu, 0 to stop playing: ")
            if int(stop) == 0:
                break 




if __name__ == "__main__":

    print("Test file is running.")

    # Load MDP file
    mdp_file = "ex_craps.mdp"
    states, actions, transitions = parse_mdp_file(mdp_file)


    states = set(state for state, _ in transitions.keys())
    print(states)

    # Automatically generate state_actions
    state_actions = {state: set() for state in states}

    for (state, action) in transitions:
        if action is not None:
            state_actions[state].add(action)

    # Convert sets to lists and handle states with no actions
    state_actions = {state: list(actions) if actions else None for state, actions in state_actions.items()}
    print(state_actions)

    G = build_graph(transitions)

    # Run the simulation


    user_input = input("Enter the start state (press Enter to use the default 'I'): ").strip()

    # Use the default value 'I' if the user input is empty
    simulate(start_state=user_input if user_input else 'I')
    #play_player()



