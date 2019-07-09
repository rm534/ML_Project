import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
import AE_new
import numpy as np
import random


class evoEA():
    def __init__(self, population_size, structure):
        self.input = [structure[0]]
        self.population_size = population_size
        self.population = self.init_population(structure)
        self.evaluate_population()
        self.encode_population()

        # create initial population of AEs
        # Evaluate fitness of all the AEs
        # encode these into genomes
        # choose parents
        # crossover, mutate (EA algorithm)
        # Evaluate fitness of new population
        # Repeat until converged

    def init_population(self, structure):
        population = []
        for i in range(0, self.population_size):
            population.append(AE_new.AutoEncoder(structure))

        return population

    def evaluate_population(self, epochs=500, training_data_len=59000, batch_size=256, data_len=60000):
        logging.info("evaluate_population - evaluating population....")
        self.fitness = []
        for element in self.population:
            element.train_model(epochs=epochs, training_data_len=training_data_len, batch_size=batch_size,
                                data_len=data_len)
            self.fitness.append(element.history.history["mean_squared_error"][-1])
        fitness_arr = np.array(self.fitness)
        self.avg_fitness = np.sum(fitness_arr)/len(self.fitness)
        logging.info("evaluate_population - average fitnes: {}".format(self.avg_fitness))
        logging.info("evaluate_population - fitness: {}".format(self.fitness))

    def encode_population(self):
        logging.info("encode_population - encoding population...")
        self.genome_population = {}
        for element in self.population:
            self.genome_population["AE" + str(self.population.index(element))] = {
                "index": self.population.index(element),
                "weights": self.get_model_weights(element.autoencoder),
                "bias": self.get_model_bias(element.autoencoder),
                "fitness": self.fitness[self.population.index(element)],
                "layers": self.get_layers(element.autoencoder)
            }

    def evaluate_children(self, epochs=500, training_data_len=5000, batch_size=256, data_len=6000):
        self.fitness_children = []
        for element in self.children:
            element.train_model(epochs=epochs, training_data_len=training_data_len, batch_size=batch_size,
                                data_len=data_len)
            fitness = element.history.history["mean_squared_error"][-1]
            self.fitness_children.append(fitness)
        i=0

        for key in self.genome_children:
            self.genome_children[key]["fitness"] = self.fitness_children[i]
            i += 1
            logging.info("evaluate_children - fitness_children: {}".format(self.genome_children[key]["fitness"]))
    def replace_worst(self):
        #self.population = np.array(self.population)
        logging.info("replace_worst - fitness: {}".format(self.fitness))

        for key in self.genome_children:
            if self.genome_children[key]["fitness"] < max(self.fitness):
                #np.put(self.population, [np.where(self.fitness == np.amin(self.fitness))], [self.genome_children[key]])
                self.population[self.fitness.index(max(self.fitness))] = self.genome_to_model(self.genome_children[key])
                self.fitness[self.fitness.index(max(self.fitness))] = self.genome_children[key]["fitness"]
        logging.info("replace_worst - fitness: {}".format(self.fitness))
        fitness_arr = np.array(self.fitness)
        self.avg_fitness = np.sum(fitness_arr) / len(self.fitness)


    def decode_children(self):
        self.children = []
        for key in self.genome_children:
            self.children.append(self.genome_to_model(self.genome_children[key]))

    def decode_population(self):
        logging.info("decode_population - decoding population...")
        self.population = []
        for i in range(0, self.population_size):
            self.population.append(self.genome_to_model(self.genome_population["AE" + str(i)]))
        #print(self.population)

    def genome_to_model(self, genome):

        model = AE_new.AutoEncoder(genome["layers"])
        #print(np.shape(model.autoencoder.layers[1].get_weights()[0]))

        for i in range(0, len(model.autoencoder.layers)):
            if i == 0:
                pass
            else:
                #print(np.shape(genome["weights"][i - 1]))
                model.autoencoder.layers[i].set_weights([genome["weights"][i - 1], genome["bias"][i - 1]])
      #  print(model)
        return model

    def choose_parent(self):
        fitness_new = []
        prob = []
        for element in self.fitness:
            element_new = 1 / element
            fitness_new.append(element_new)
        sum = np.sum(fitness_new)
        for element in fitness_new:
            prob.append(element / sum)

        par_choice = np.random.choice(self.population, p=prob)
        return self.population.index(par_choice)

    def choose_parents(self, parent_num):
        logging.info("choose_parents - choosing parents...")
        fitness_new = []
        prob = []
        for element in self.fitness:
            element_new = 1 / element
            fitness_new.append(element_new)
        sum = np.sum(fitness_new)
        for element in fitness_new:
            prob.append(element / sum)
        parents = []
       # print(prob)
        for i in range(0, parent_num):
            par_choice1 = np.random.choice(self.population, p=prob)
            par_choice2 = np.random.choice(self.population, p=prob)
            parents.append((self.population.index(par_choice1), self.population.index(par_choice2)))
       # print(parents)
        logging.debug("choose_parents - parents: {}".format(parents))
        return parents

    def get_weights_front(self, layer, genome, node, arr_elements = True):
        weights_front = []
        if arr_elements == True:
            logging.debug("get_weights_front - shape of front weights at layer {}: {}".format(layer, np.shape(genome["weights"][layer-1])))
            for element in genome["weights"][layer - 1]:
                weights_front.append([element[node]])
        elif arr_elements == False:
            logging.debug("get_weights_front - shape of front weights at layer {}: {}".format(layer, np.shape(
                genome["weights"][layer - 1])))
            for element in genome["weights"][layer - 1]:
                weights_front.append(element[node])

        return np.array(weights_front)

    def get_weights_back(self, layer, genome, node):
       # print(type(genome["weights"][-layer - 1]))
        weights_back = np.array(genome["weights"][layer][node])
        logging.debug("get_weights_back - shape of back weights at layer {}: {}".format(layer, np.shape(genome["weights"][layer])))
       # print(weights_back)
        return weights_back

    def shape_correction_crossover(self, weight_set1, weight_set2):
        if np.shape(weight_set1)[0] < np.shape(weight_set2)[0]:
            rand_choice_arr = []
            diff = np.shape(weight_set2)[0] - np.shape(weight_set1)[0]
            for i in range(0, diff):
                rand_choice = random.randint(0, np.shape(weight_set1)[0]-1)
                rand_choice_arr.append(weight_set1[rand_choice])
            weight_set1 = np.append(weight_set1, rand_choice_arr)
        elif np.shape(weight_set2)[0] < np.shape(weight_set1)[0]:
            rand_choice_arr = []
            diff = np.shape(weight_set1)[0] - np.shape(weight_set2)[0]
            for i in range (0, diff):
                rand_choice = random.randint(0, np.shape(weight_set1)[0]-1)
                rand_choice_arr.append(weight_set2[rand_choice])
            weight_set2 = np.append(weight_set2, rand_choice_arr)
        else:
            pass
        return weight_set1, weight_set2

    def cross_over_weights(self, weights_front1, weights_front2, weights_back1, weights_back2, node, genome1, genome2, layer):

        for i in range(0, len(genome1["weights"][layer-1])):

            logging.debug("cross_over_weights - front weights parent 1: {} front weights parent 2: {} crossover node: {} crossover value: {}".format(genome1["weights"][layer - 1][i], genome2["weights"][layer - 1][i], node, genome2["weights"][layer - 1][i][node]))
            #logging.debug("cross_over_weights - front weights before crossover at layer {}: {}".format(layer, genome1["weights"][layer - 1][i]))
            np.put(genome1["weights"][layer-1][i], [node], [weights_front2[i]])
            logging.debug("cross_over_weights - front weights after crossover at layer {}: {}".format(layer, genome1["weights"][layer - 1][i]))

            np.put(genome2["weights"][layer-1][i], [node], [weights_front1[i]])
        np.put(genome1["weights"][layer], [node], [weights_back2])
        np.put(genome2["weights"][layer], [node], [weights_back1])

    def layer_compare(self, layer1, layer2):
        if layer1 < layer2:
            return layer1
        elif layer2 < layer1:
            return layer2
        else:
            return layer1

    def create_children(self, parents):
        self.genome_children = {}
        for parent in parents:
            for item in parent:
                self.genome_children["AE" +str(parents.index(parent))+ str(item)] = {
                    "index": parents.index(parent),
                    "weights": self.genome_population["AE"+str(parent[0])]["weights"],
                    "bias": self.genome_population["AE"+str(parent[0])]["bias"],
                    "fitness": 0,
                    "layers": self.genome_population["AE"+str(parent[0])]["layers"]
                }

    def crossover(self, parents):
        self.create_children(parents)
        logging.info("crossover - crossing over between parents...")
        for parent in parents:
            layers_1 = self.genome_population["AE" + str(int(parent[0]))]["layers"]
            layers_2 = self.genome_population["AE" + str(int(parent[1]))]["layers"]

            # Choose random amount of nodes to crossover and which nodes
            for i in range(0, len(layers_1)-1):
                if i == 0:
                    pass
                else:
                    crossover_len = random.randint(0, random.randint(0, self.layer_compare(layers_1[i],layers_2[i]) - 1))
                    crossover_nodes = []

                    for j in range(0, crossover_len):
                        crossover_nodes.append(random.randint(0, self.layer_compare(layers_1[i],layers_2[i]) - 1))
                    if not crossover_nodes:
                        return

                    logging.debug("crossover - crossover_nodes:{}".format(crossover_nodes))
                    #self.get_weights_front(i, self.genome_population["AE"+str(int(parent[0]))], crossover_nodes[j])
                    for j in range(0, crossover_len):
                        weights1_front = self.get_weights_front(i, self.genome_population["AE" + str(int(parent[0]))], crossover_nodes[j], arr_elements=False)
                        weights2_front = self.get_weights_front(i, self.genome_population["AE" + str(int(parent[1]))], crossover_nodes[j], arr_elements=False)
                      #  print("weights_front2: {}".format(weights2_front))
                        weights1_back = self.get_weights_back(i, self.genome_population["AE" + str(int(parent[0]))],  crossover_nodes[j])
                        weights2_back = self.get_weights_back(i, self.genome_population["AE" + str(int(parent[1]))],  crossover_nodes[j])
                        weights1_front, weights2_front = self.shape_correction_crossover(weights1_front, weights2_front)
                        weights1_back, weights2_back = self.shape_correction_crossover(weights1_back, weights2_back)
                        self.cross_over_weights(weights1_front, weights2_front, weights1_back, weights2_back, crossover_nodes[j], self.genome_children["AE"+str(parents.index(parent))+str(parent[0])], self.genome_children["AE"+str(parents.index(parent))+str(parent[1])], i)
                      #  print("genome1 shape: {}".format(self.genome_population["AE"+str(int(parent[0]))]["weights"][-i - 2]))

        logging.debug("crossover - shape of weight matrices: {}".format(np.shape(self.genome_population["AE" + str(int(parent[0]))]["weights"][0])))
        logging.debug("crossover - shape of layers: {}".format(self.genome_population["AE" + str(int(parent[0]))]["layers"]))

    def add_new_node(self, genome_add, genome_new, layer):
        layers_arr_add = genome_add["layers"]
        layer_arr_new = genome_new["layers"]
        logging.debug("add_new_node - layers shize at layer {}: {}".format(layer-1, layers_arr_add[layer-1]))
        logging.debug("add_new_node - shape of bias at layer {}: {}".format(layer-1, np.shape(genome_add["bias"][layer])))
        node_choice = random.randint(0, layer_arr_new[layer] - 1)
        node_choice_bias =random.randint(0, layers_arr_add[layer-1] - 1)
        new_weights_front = []
       # print(genome_new["bias"][-layer - 2])
        # print("weights:{}".format(genome_new["weights"][-layer-1]))
        # print("layer:{}".format(layer_arr_new[layer]))
        for element in genome_new["weights"][layer - 1]:
            #    print(element[node_choice])
            new_weights_front.append([element[node_choice]])

        new_weights_front = self.get_weights_front(layer, genome_new, node_choice)
        #new_weights_front = np.array(new_weights_front)
        # print(new_weights_front)
        # print(len(genome_new["weights"]))
        #new_weights_back = genome_new["weights"][layer][node_choice]
        new_weights_back = self.get_weights_back(layer, genome_new, node_choice)
        new_bias_back = genome_new["weights"][layer][node_choice]
      #  print(new_weights_back)
        logging.debug("add_new_node - shape of genome_add weights: {}".format(np.shape(genome_add["weights"][layer - 1])))
        logging.debug("add_new_node - shape of new_weights_front: {}".format(np.shape(new_weights_front)))
        logging.debug(
            "add_new_node - shape of genome_add weights: {}".format(np.shape(genome_add["weights"][layer])))
        logging.debug("add_new_node - shape of new_weights_back: {}".format(np.shape(new_weights_back)))
        genome_add["weights"][layer - 1] = np.append(genome_add["weights"][layer - 1], new_weights_front, axis=1)
        genome_add["bias"][layer - 1] = np.append(genome_add["bias"][layer - 1],
                                                   [genome_add["bias"][layer][node_choice_bias]], axis=0)
        genome_add["weights"][layer] = np.append(genome_add["weights"][layer], [new_weights_back], axis=0)
      #  print(genome_add["weights"][-layer - 1])
        # print(genome_add["weights"][-layer-2] )
        genome_add["layers"][layer] += 1

    def delete_node(self, genome_delete, layer):
        layer_arr_new = genome_delete["layers"]
        node_choice = random.randint(0, layer_arr_new[layer] - 1)
       # print(genome_delete["bias"])
        genome_delete["weights"][layer - 1] = np.delete(genome_delete["weights"][layer - 1], [node_choice], axis=1)
        genome_delete["weights"][layer] = np.delete(genome_delete["weights"][-layer], [node_choice], axis=0)
        genome_delete["bias"][layer - 1] = np.delete(genome_delete["bias"][layer - 1], [node_choice], axis=0)
       # print(genome_delete["weights"][-layer - 2])
        genome_delete["layers"][layer] -= 1

    def mutate(self, parents):
        logging.info("mutate - mutating population")
        for parent in parents:
            layers_1 = self.genome_population["AE" + str(int(parent[0]))]["layers"]
            layers_2 = self.genome_population["AE" + str(int(parent[1]))]["layers"]
            parent1_weights = self.genome_population["AE" + str(int(parent[0]))]["weights"]
            parent2_weights = self.genome_population["AE" + str(int(parent[1]))]["weights"]
            prob = [0.5, 0.5]
            mutate_prob = [True, False]
            for i in range(0, len(parent)):
                for j in range(0, len(layers_1)-1):
                    # skip input layer  
                    if j == 0:
                        pass
                    else:
                        mutate = np.random.choice(mutate_prob, p=prob)
                       # print(mutate)

                        if mutate == True:
                            #print(i)

                            if i < len(layers_1):
                                self.add_new_node(self.genome_children["AE"+str(parents.index(parent))+str(parent[i])],
                                                  self.genome_population["AE" + str(int(self.choose_parent()))], j)
                            elif i >= len(layers_1):
                                pass
                        elif mutate == False:
                            self.delete_node(self.genome_children["AE"+str(parents.index(parent))+str(parent[i])], j)

           # print("AE0 layers:{}".format(self.genome_population["AE0"]["layers"]))
           # print("AE0 weights:{}".format(np.shape(self.genome_population["AE0"]["weights"][0])))
           # print("AE1 layers:{}".format(self.genome_population["AE1"]["layers"]))
           # print("AE1 weights:{}".format(np.shape(self.genome_population["AE1"]["weights"][1])))
           # print("AE2 layers:{}".format(self.genome_population["AE2"]["layers"]))
           # print("AE2 weights:{}".format(np.shape(self.genome_population["AE2"]["weights"][1])))

    def evolve(self, parent_num):
        parents = self.choose_parents(parent_num=parent_num)
        self.crossover(parents)
        self.mutate(parents)
        self.decode_children()
        self.evaluate_children()
        self.replace_worst()

    def optimise(self, epochs, parent_num=1):
        self.history = {
            "average fitness": [],
            "best performing individual fitness": [],

        }
        for i in range(0, epochs):
            self.history["average fitness"].append(self.avg_fitness)
            self.history["best performing individual fitness"].append(min(self.fitness))
            self.evolve(parent_num=parent_num)
        logging.info("optimise - average fitness: {}".format(self.history["average fitness"]))
        logging.info("optimise - best fitness: {}".format(self.history["best performing individual fitness"]))

    def get_model_weights(self, model):
        weights = []
        for layer in model.layers:

            if model.layers.index(layer) == 0:
                pass
            else:
                weights.append(layer.get_weights()[0])
        return weights

    def get_model_bias(self, model):
        bias = []
        for layer in model.layers:
            if model.layers.index(layer) == 0:
                pass
            else:
                bias.append(layer.get_weights()[1])
        return bias

    def get_layers(self, model):
        layers = []
        for element in model.layers:
            if model.layers.index(element) == 0:
                pass
            else:
                layers.append(element.get_config()["units"])
        layers = self.input + layers
        logging.debug("get_layers - layers: {}".format(layers))
        return layers


if __name__ == "__main__":
    EA = evoEA(30, [18,512,18])
    EA.optimise(1000, parent_num=12)
