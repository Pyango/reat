import gym
import numpy as np

from entities.activation import relu_activation
from entities.population import Population
import zmq

population = Population(
    num_inputs=4,
    num_outputs=2,
    fitness_threshold=500,  # frames how long it should survive
    initial_fitness=0,
    survival_threshold=3,
    compatibility_threshold=1,
    max_species=10,
    size=150,
    output_activation_function=relu_activation,
)

runs_per_net = 2


def compute_fitness(genomes):
    for genome_key, genome in genomes:
        fitnesses = []

        for runs in range(runs_per_net):
            env = gym.make("CartPole-v1")

            observation = env.reset()
            fitness = 0.0
            done = False
            while not done:
                action = np.argmax(genome.activate(observation))
                observation, reward, done, info = env.step(action)
                fitness += reward

            fitnesses.append(fitness)
        genome.fitness = np.mean(fitnesses)


def on_success(best):
    env = gym.make("CartPole-v1")
    observation = env.reset()
    done = False
    while not done:
        action = np.argmax(best.activate(observation))
        observation, reward, done, info = env.step(action)
        env.render()

import json

context = zmq.Context()

socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# population.run(compute_fitness, on_success)
env = gym.make("CartPole-v1")

observation = env.reset()
fitness = 0.0
done = False
while not done:
    observation, reward, done, info = env.step(1)
    socket.send(json.dumps(observation.tolist()).encode())
    message = socket.recv()
    print(f"reversed: {message}")

    print(observation, reward, done, info)

