import numpy as np
import tensorflow as tf
from agent import Agent
import copy
from matplotlib import pyplot as plt
import time
import networkx as nx
plt.ion()

parameters = {"innoise" : 2, # Stddev on incomming messages
              "outnoise" : 2, # Stddev on outgoing messages
              "num_environment" : 5, # Num univariate environment nodes
              "num_agents" : 10, # Number of Agents
              "fanout" : 1, # Distinct messages an agent can say
              "statedim" : 1, # Dimension of Agent State
              "envnoise": 25, # Stddev of environment state
              "envobsnoise" : 2, # Stddev on observing environment
              "batchsize" : 1000} # Training Batch Size

tf.set_random_seed(634)
class Organization(object):
    def __init__(self, num_environment, num_agents, innoise,
                     outnoise, fanout, statedim, envnoise, envobsnoise,
                     batchsize, **kwargs):
        self.num_environment = num_environment
        self.batchsize = batchsize
        self.envobsnoise = envobsnoise
        self.agents = []
        for i in range(num_agents):
            self.agents.append(Agent(innoise, outnoise, i, fanout, statedim, batchsize))
        self.environment = tf.random_normal([self.batchsize, num_environment],
                                            mean=0, stddev = envnoise, dtype=tf.float64)
        self.build_org()
        self.objective  =  self.loss()
        self.learning_rate = tf.placeholder(tf.float64)
        self.decay= 0.001
        #self.optimize = tf.train.AdadeltaOptimizer(self.learning_rate, rho=.9).minimize(self.objective)
        #self.optimize = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.objective)
        self.optimize = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.objective)
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def build_org(self):
        self.build_agent_params()
        self.build_wave()
        

    def build_agent_params(self):
        indim = self.num_environment
        for ix, a in enumerate(self.agents):
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
#            print innoise, indata, a.listen_weights
            noisyin = indata  +  innoise/a.listen_weights
            #noisyin = indata * a.listen_weights + innoise
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
        #difference = tf.reduce_mean(self.environment) - self.agents[-1].state
        differences = [tf.reduce_mean((tf.reduce_mean(self.environment, 1, keep_dims=True) - a.state)**exponent) for a in self.agents]
        differences = tf.add_n(differences)
        cost  = self.listening_cost() + self.speaking_cost()
        loss =  -(-differences - cost)
        return loss

    def train(self, niters, lrinit=.00001, iplot=False, verbose=False):
        if iplot:
            fig, ax = plt.subplots()
            ax.plot([1],[1])
            ax.set_xlim(0,niters)
            ax.set_ylim(0,10)
            ax.set_ylabel("Welfare (Log)")
            ax.set_xlabel("Training Epoch")
            line = ax.lines[0]
        training_res = []
        for i  in range(niters):
            lr = lrinit / (1+ i*self.decay)
            self.sess.run(self.optimize, feed_dict={self.learning_rate:lr})
            strat = self.sess.run(self.agents[0].listen_weights)
            u = self.sess.run(self.objective)
            if verbose:
                print  u
            training_res.append(u)
            if (i%50==0) and iplot:
                line.set_data(np.arange(len(training_res)), np.log(training_res))
                fig.canvas.draw()
        listen_params = self.sess.run([a.listen_weights for a in self.agents])
        return Results(training_res, listen_params)
    
    def reset(self):
        for agent in self.agents:
            assignments = agent.random_reset()
            self.sess.run(assignments)

class Results(object):
    def __init__(self, training_res, listen_params):
        self.training_res = training_res
        self.listen_params = listen_params
        self.get_trimmed_listen_params()

    def get_trimmed_listen_params(self, cutoff=.1):
        self.trimmed = []
        for lparams in self.listen_params:
            maxp = np.max(lparams)
            lparams = lparams * np.int_(lparams * lparams>cutoff*maxp)
            self.trimmed.append(lparams)

    def graph_org(self, vspace=1, hspace=2):
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
        colors = nx.get_node_attributes(G, "color").values()
        pos= nx.get_node_attributes(G, "pos")
        nx.draw(G, pos, node_color = colors, with_labels=True,
                    labels=nx.get_node_attributes(G, "name"), alpha=.5, node_size=600 )
        return G
            
    def _get_pos(self, G):
        numenv = len(self.trimmed[0].flatten())
        numnodes = numenv + len(self.trimmed)
        
    
    
        
org = Organization(**parameters)
org.train(1000, verbose=True)
