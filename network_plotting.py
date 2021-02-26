import pandas as pd


import networkx as nx
import itertools
import numpy as np

import plotly.offline as py
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler


def scale_edge_weights(graph):
    """
    Scale the edge weights of a networkx graph for graphing and return the new graph.
    """
    g = graph.copy()
    original_weights = []
    for edge in g.edges():
        original_weights.append(g.edges()[edge]['weight'])
    scaler = MinMaxScaler(feature_range=(.5,12))
    new_weights = scaler.fit_transform(np.array(original_weights).reshape(-1,1)).flatten()
    for i,edge in enumerate(g.edges()):
        g.edges()[edge]['weight'] = new_weights[i]
    return g


def make_edges(graph, pos, unscaled,show_all, set_width):
    """
    Return go.Scatter traces representing the edges and the midpoints of those edges (which have captions) for a networkx graph
    Args:

    graph: networkx graph
    pos: postions of nodes
    unscaled: the unscaled version of graph, for captions
    show_all (boolean): Whether or not to show all connections regardless of size
    set_width (None or int): If not None, set the width of all the visible edges to be set_width

    """

    edge_traces = []
    edge_text_xs = []
    edge_text_ys = []
    edge_text_labels = []
    for edge in graph.edges():
        width = graph.edges()[edge]['weight']

        if width < .6 and show_all is False:
            continue
        if set_width is not None:
            width = set_width
        #Make it so smaller edges are more transparent. These numbers are a bit random, I jusst played wit them until they looked good.
        transparency = max(.5,round(width/5,2))


        #royalblue
        color_string = f'rgba(65, 105, 225, {transparency})'

        char1  = edge[0]
        char2  = edge[1]
        x0, y0 = pos[char1]
        x1, y1 = pos[char2]

        x = [x0, x1, None]
        y = [y0, y1, None]

        #Add edges (i.e. actual lines that appear)
        edge_trace = go.Scatter(x     = x,
                                y     = y,
                                line  = dict(width = width,
                                             color = color_string),
                                mode  = 'lines')
        edge_traces.append(edge_trace)

        #Calculate midpoints, get the number of conenctions that should be displayed
        edge_text_xs.append((x0+x1)/2)
        edge_text_ys.append((y0+y1)/2)
        connections = unscaled.edges()[edge]['weight']
        edge_text_labels.append(char1.capitalize() + ' -- ' + char2.capitalize() + f': {connections} connections')

    #Add midpoint text trace
    edge_text_trace = go.Scatter(x         = edge_text_xs,
                                 y         = edge_text_ys,
                                 text      = edge_text_labels,
                                 textposition = "bottom center",
                                 textfont_size = 10,
                                 mode      = 'markers',
                                 hoverinfo = 'text',
                                 marker    = dict(color = 'rgba(0,0,0,0)',
                                                 size  = 1,
                                                 line  = None))

    return edge_traces, edge_text_trace

def plot_network(graph, chars = None, show_all = False, set_width = None, output='plot'):
    """
    Plot/return a networkx network nicely

    Args:
    graph: networkx graph
    chars (None or list of str): If not None then only graph a network of the characters in chars. Should be smaller than and only include strings in the list used to make the graph, for now important_chars

    show_all (bool): Whether or not to show all edges regardless of size/number of connections
    set_width (None or int): If not None, set the width of all the visible edges to be set_width
    output: If 'plot', plots output. If 'return', returns the figure (used in streamlit app). If 'save', saves an image. If none of those, also just plots it.


    """
    if chars is not None:
        graph = graph.subgraph(chars)

    scaled = scale_edge_weights(graph)
    pos = nx.spring_layout(graph, k =.75 , seed = 1)

    #Add edges
    edge_traces, edge_text_trace = make_edges(scaled, pos, graph, show_all, set_width)

    #Add nodes
    node_xs = [pos[node][0] for node in scaled.nodes()]
    node_ys = [pos[node][1] for node in scaled.nodes()]
    node_text = ['<b>'+node.capitalize() for node in scaled.nodes()]
    node_hovertext = []
    for node in graph.nodes():
        node_hovertext.append(node.capitalize() + ': '+  str(graph.nodes()[node]['size']) + ' appearances')
    node_trace = go.Scatter(x     = node_xs,
                        y         = node_ys,
                        text      = node_text,
                        textposition = "bottom center",
                        textfont_size = 14,
                        mode      = 'markers+text',
                        hovertext = node_hovertext,
                        hoverinfo = 'text',
                        marker    = dict(color = 'black',#'#6959CD',
                                         size  = 15,
                                         line  = None))
    layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig = go.Figure(layout = layout)

    for trace in edge_traces:
        fig.add_trace(trace)
    fig.add_trace(node_trace)
    fig.add_trace(edge_text_trace)

    fig.update_layout(showlegend = False, width = 1000, height = 1200)
    fig.update_xaxes(showticklabels = False)
    fig.update_yaxes(showticklabels = False)

    if output == 'plot':
        fig.show()
    elif output == 'return':
        return fig
    elif output == 'save':
        fig.write_image('graph.png')
    else:
        fig.show()
