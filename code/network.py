#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from agent import Agent
import copy
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import time
import networkx as nx
plt.ion()

class Organization(object):
    def __init__(self, num_environment, num_agents, innoise,
                     outnoise, fanout, statedim, envnoise, envobsnoise,
                     batchsize, optimizer, randomSeed=False, **kwargs):
        if( randomSeed == False ):
            tf.set_random_seed(634)
        self.num_environment = num_environment
        self.batchsize = batchsize
        self.envobsnoise = envobsnoise
        self.agents = []
        for i in range(num_agents):
            self.agents.append(Agent(innoise, outnoise, i, fanout, statedim, batchsize, num_agents))
        self.environment = tf.random_normal([self.batchsize, num_environment],
                                            mean=0, stddev = envnoise, dtype=tf.float64)
        self.build_org()
        self.objective  =  self.loss()
        self.learning_rate = tf.placeholder(tf.float64)

        # Justin used the "AdadeltaOptimizer"
        optimizers = {
            "momentum":         tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9).minimize(self.objective),
            "adadelta":         tf.train.AdadeltaOptimizer(self.learning_rate, rho=.9).minimize(self.objective),
            "adam":             tf.train.AdamOptimizer(self.learning_rate).minimize(self.objective),
            "rmsprop":          tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.objective),
            "gradient-descent": tf.train.AdagradOptimizer(self.learning_rate).minimize(self.objective)
            }

        learning_rates = {
            "momentum":         0.00001,
            "adadelta":         100,
            "adam":             1e-12,
            "rmsprop":          1e-12,
            "gradient-descent": 0.00001
        }

        decays = {
            "momentum":         None,
            "adadelta":         0.001,
            "adam":             None,
            "rmsprop":          None,
            "gradient-descent": 0.001
        }

        self.optimize = optimizers[optimizer]
        self.start_learning_rate = optimizers[optimizer]
        self.decay = decays[optimizer]

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def build_org(self):
        self.build_agent_params()
        self.build_wave()
        

    def build_agent_params(self):
        indim = self.num_environment
        for ix, a in enumerate(self.agents):
            print "Agent %d gets indim=%d" % (ix, indim)
            a.create_in_vec(indim)
            a.create_state_matrix(indim)
            a.create_out_matrix(indim)
            indim += a.fanout

    def build_wave(self):
        """
        This loops through agents building the tensorflow objects
        that determine the agents states and outputs that are then
        recursively use to build all other agent's states and outputs
        """
        self.states = []
        self.outputs = []
        for a in self.agents:
            envnoise = tf.random_normal([self.batchsize, self.num_environment], stddev=self.envobsnoise, dtype=tf.float64)
            #envnoise = tf.random_uniform([self.batchsize, self.num_environment],
                                            #minval = -self.envobsnoise, maxval= self.envobsnoise, dtype=tf.float64)
            inenv = self.environment
            incomm = None #?
            for inmsgs in self.outputs:
                if incomm is None:
                    incomm = inmsgs # Stays None if inmsgs blank, otherwise becomes inmsgs
                else:
                    incomm =  tf.concat([incomm, inmsgs], 1) # If already a message, then concat
            commnoise = tf.random_normal([self.batchsize, a.indim - self.num_environment], stddev=a.noiseinstd, dtype=tf.float64)
            #commnoise = tf.random_uniform([self.batchsize, a.indim - self.num_environment],
                                              #minval=a.noiseinstd, maxval=a.noiseinstd, dtype=tf.float64)
            # Noise on inputs
            if incomm is not None:
                indata = tf.concat([inenv, incomm], 1) # batchsize x 
            else:
                indata = inenv
            innoise = tf.concat([envnoise, commnoise], 1)
            #print innoise, indata, a.listen_weights

            # Add noise inversely-proportional to listening strength
            noisyin = indata  +  innoise/a.listen_weights

            # Since listen weights is 1xin we get row wise division.
            state = tf.matmul(noisyin, a.state_weights)
            a.state = state
            self.states.append(state)

            outnoise = tf.random_normal([self.batchsize, a.fanout], stddev=a.noiseoutstd, dtype=tf.float64)
            #outnoise = tf.random_uniform([self.batchsize, a.fanout], minval=a.noiseoutstd,maxval=a.noiseoutstd, dtype=tf.float64)
            prenoise = tf.matmul(noisyin, a.out_weights)
            output = prenoise + outnoise
            self.outputs.append(output)

    def listening_cost(self, exponent=2):
        summed = [tf.reduce_sum(tf.abs(x.listen_weights))**exponent for x in self.agents]
        totalc = tf.add_n(summed)
        return totalc

    def speaking_cost(self, exponent=2):
        summed = [tf.reduce_sum(tf.abs(x.out_weights))**exponent for x in self.agents]
        totalc = tf.add_n(summed)
        return totalc

    def loss(self, exponent=2):
        # Get the difference^2 of how far each agent is from real avg of variables
        realValue = tf.reduce_mean(self.environment, 1, keep_dims=True)
        differences = [tf.reduce_mean((realValue - a.state)**exponent) for a in self.agents]
        differences = tf.add_n(differences)
        cost = self.listening_cost() + self.speaking_cost()
        loss = differences + cost
        # Restrict output to 0..200
        #normalized_loss = tf.multiply(tf.sigmoid(loss), tf.constant(200.0, dtype=tf.float64))
        #return tf.minimum(loss, tf.constant(200.0, dtype=tf.float64))
        return loss

    def train(self, niters, lrinit=None, iplot=False, verbose=False):
        if( lrinit == None ):
            lrinit = self.start_learning_rate
        if iplot:
            fig, ax = plt.subplots()
            ax.plot([1],[1])
            ax.set_xlim(0,niters)
            ax.set_ylim(0,10)
            ax.set_ylabel("Welfare (Log)")
            ax.set_xlabel("Training Epoch")
            line = ax.lines[0]
        training_res = []

        # For each iteration
        for i  in range(niters):

            # Run training, and adjust learning rate if it's an Optimizer that
            # works with decaying learning rates (some don't)
            lr = lrinit
            if( self.decay != None ):
            	lr = lrinit / (1 + i*self.decay) # Learn less over time
            self.sess.run(self.optimize, feed_dict={self.learning_rate:lr})

            #for a in self.agents:
                #a.normalize()
            listen_params = self.sess.run([a.listen_weights for a in self.agents])
            if verbose:
                print "Listen_params now set to: " + str(listen_params)

            # Prints the agent's current strategy at each step so we can see how well it's doing
            #strat = self.sess.run(self.agents[0].listen_weights)
            #print(strat)

            # Evaluates our current progress towards objective
            u = self.sess.run(self.objective)
            if verbose:
                print  "Loss function=" + str(u)
            training_res.append(u)

            if (i%50==0) and iplot:
                line.set_data(np.arange(len(training_res)), np.log(training_res))
                fig.canvas.draw()

        # Get the strategy from all agents, which is the "network configuration" at the end
        listen_params = self.sess.run([a.listen_weights for a in self.agents])
        print "Listen_params now set to: " + str(listen_params)
        return Results(training_res, listen_params)
    
class Results(object):
    def __init__(self, training_res, listen_params):
        self.training_res = training_res
        self.listen_params = listen_params
        self.get_trimmed_listen_params()

    def get_trimmed_listen_params(self, cutoff=.1):
        self.trimmed = []
        for lparams in self.listen_params:
            maxp = np.max(lparams)
            print str(lparams)
            # Line below is where most optimizing functions feak out
            lparams = lparams * np.int_(lparams * lparams>cutoff*maxp)
            self.trimmed.append(lparams)

    def generate_graph(self, vspace=1, hspace=2):
        numenv = len(self.trimmed[0].flatten())
        numnodes = numenv + len(self.trimmed)
        G = nx.DiGraph()
        vpos = 0
        hpos = 0
        for i in range(numenv):
            G.add_node(i, color="b", name="E" + str(i), pos=(hpos, vpos))
            hpos += hspace
        vpos = vspace
        hpos = 0
        hspace = hspace*numenv/float(len(self.trimmed))
        highest_listened = numenv - 1
        for aix, agent in enumerate(self.trimmed):
            hpos += hspace
            nextlevel = False
            nodenum = numenv +aix
            G.add_node(nodenum, color='r', name = "A" + str(aix))
            for eix, val in enumerate(agent.flatten()):
                if abs(val) > 0:
                    G.add_edge(eix, nodenum, width=val)
                    if eix > highest_listened:
                        highest_listened =eix
                        nextlevel=True
            if nextlevel:
                vpos += vspace
            G.node[nodenum]["pos"] = (hpos, vpos)
        return G

    def graph_org(self, vspace=1, hspace=2):
        G = self.generate_graph(vspace, hspace)
        colors = nx.get_node_attributes(G, "color").values()
        pos= nx.get_node_attributes(G, "pos")
        nx.draw(G, pos, node_color = colors, with_labels=True,
                    labels=nx.get_node_attributes(G, "name"), alpha=.5, node_size=600 )
        return G

    def graph_cytoscape(self, filename, vspace=1, hspace=2):
        numenv = len(self.trimmed[0].flatten())
        G = nx.DiGraph()
        for i in range(numenv):
            G.add_node(i, color="b", name="E" + str(i), category="environment")
        for aix, agent in enumerate(self.trimmed):
            nodenum = int(numenv + aix)
            G.add_node(nodenum, color='r', name="A" + str(aix), category="agent")
            # For each node, weights will be zero if the edge should be ignored
            # and otherwise represent the cost of the edge
            for dest, weight in enumerate(agent.flatten()):
                if( abs(weight) > 0 ):
                    G.add_edge(int(dest), nodenum, width=float(weight), weight=float(abs(weight)))
        nx.write_graphml(G, filename)
        #nx.write_gml(G, filename)
            
    def _get_pos(self, G):
        numenv = len(self.trimmed[0].flatten())
        numnodes = numenv + len(self.trimmed)
