__author__ = "Robin M."
import random
import wntr
import networkx as nx
import numpy as np
import math
import logging
import time
import pandas as pd


logging.basicConfig(level=logging.INFO)

PATTERN_HIGH = 3.5
PATTERN_LOW = 0.2
MEASUREMENT_NODES = ["40954616_28569053", "40942531_28559853", "40918053_28577084", "40900894_28615759",
                     "40918044_28600378", "40937953_28551144", "40887069_28587966", "40905200_28563978",
                     "40912078_28559763", "40906741_28582519", "40937216_28597569", "40960499_28594427",
                     "40956184_28551447", "40953103_28539756", "40890628_28614094", "40913916_28611584",
                     "40969684_28543009", "40975098_28575131"]
MEASUREMENT_NODES_2 = ["39805881_29979691", "39815944_29974153", "39782531_30006725",
                       "39767331_29942488", "39754031_29936053", "39769016_29906763", "39746197_29931584",
                       "39797059_29965881", "39857788_29954363", "39837163_29991522",
                       "39836584_29934944", "39787281_29949863", "39820650_29954556", "39807703_30006388",
                       "39777403_29925444", "39835328_29969309", "39835202_29985195"]


class NetworkGenerator():
    def __init__(self, nodes, alpha):
        self.nodes, self.alpha = nodes, alpha

    def initialise_graph(self):
        # Method to initialise all graph information

        if self.alpha <= 1 and self.alpha >= 2 / (self.nodes - 1):
            links = ((self.nodes ** 2 - self.nodes) * self.alpha) / 2

            graph_location = self.initialise_graph_location()
            graph_distance = self.initialise_graph_distance(graph_location)
            graph_patterns = self.initialise_patterns(self.nodes, 24 * 60)

            return graph_distance, graph_location, graph_patterns

    def initialise_graph_location(self):
        # Method to initialise a random location matrix

        graph_location = np.zeros((self.nodes, 2))

        for index in np.ndindex(graph_location.shape):
            # assign random location
            graph_location[index[0]][index[1]] = round(np.random.normal(0.0, 1), 2)

        logging.log(logging.DEBUG, "location matrix:\n{}".format(graph_location))
        return graph_location

    def initialise_graph_distance(self, graph_location):
        # Method to initialise a random distance matrix 

        graph_distance = np.zeros((self.nodes, self.nodes))
        for index in np.ndindex(graph_distance.shape):
            if (index[0] == index[1]):
                pass
            elif (index[0] > index[1]):
                graph_distance[index[0]][index[1]] = graph_distance[index[1]][index[0]]
            else:
                diff_x = graph_location[index[0]][0] - graph_location[index[1]][0]
                diff_y = graph_location[index[0]][1] - graph_location[index[1]][1]
                diff_arr = [diff_x, diff_y]

                graph_distance[index[0]][index[1]] = round(self.distance(diff_arr), 2)
        logging.log(logging.DEBUG, "distance matrix:\n{}".format(graph_distance))
        return graph_distance

    def distance(self, location):
        # Method to calculate distance from two location points in the graph

        distance = math.sqrt(location[0] ** 2 + location[1] ** 2)
        return distance

    def initialise_patterns(self, nodes, timestep):
        # Method to initialise pattern matrix
        pattern_length = int(round(24 * 3600 / timestep))
        graph_pattern = np.zeros((nodes, nodes, pattern_length))
        for index in np.ndindex(graph_pattern.shape):
            if index[0] == index[1]:
                pass
            elif index[0] > index[1]:
                graph_pattern[index[0]][index[1]] = graph_pattern[index[1]][index[0]]
            else:
                graph_pattern[index[0]][index[1]][index[2]] = round(np.random.uniform(PATTERN_LOW, PATTERN_HIGH), 2)

        logging.log(logging.DEBUG, "pattern matrix:\n{}".format(graph_pattern))
        return graph_pattern

    def create_network_graph(self):
        # Method to create a complete matrix with all information for creating the model
        # getting distance, location and pattern matrices
        graph_distance, graph_location, graph_patterns = self.initialise_graph()
        logging.log(logging.DEBUG,
                    "shapes:\n{}\n{}\n{}".format(graph_distance.shape, graph_location.shape, graph_patterns.shape))
        # creating a network matrix, which represents the WDN and adds all matrices together
        graph_network = np.zeros((self.nodes, self.nodes, 3 + len(graph_patterns[0][0])))
        logging.log(logging.DEBUG, "graph network shape: {}".format(graph_network.shape))
        for index in np.ndindex(graph_network.shape):

            if index[0] == index[1]:
                pass
            elif index[0] > index[1]:
                graph_network[index[0]][index[1]] = graph_network[index[1]][index[0]]
            elif index[2] == 0:
                graph_network[index[0]][index[1]][index[2]] = graph_distance[index[0]][index[1]]
            elif index[2] > 0 and index[2] < 3:
                graph_network[index[0]][index[1]][index[2]] = graph_location[index[0]][index[2] - 1]
            elif index[2] > 2:
                graph_network[index[0]][index[1]][index[2]] = graph_patterns[index[0]][index[1]][index[2] - 3]

        logging.debug("graph matrix: \n{}".format(graph_network))
        return graph_network

    def create_network_model(self, graph_network, network_name):
        # TODO: - pipe connection between node0 and node 0, node 9 and node 9
        # TODO: - first pattern is zero matrix...correct
        # TODO: - no reservoire or other elements, think how to incorporate that

        # Method to create a complete EPANET model with the complete network graph
        wn = wntr.network.WaterNetworkModel()
        # loop to initialise patterns and nodes
        for i in range(0, len(graph_network[0])):
            print(i)
            print(graph_network[i][0][3:63].tolist())
            wn.add_pattern("pat" + str(i), graph_network[i][0][3:63].tolist())
            wn.add_junction("node" + str(i), base_demand=0.01, demand_pattern="pat" + str(i))
        # loop to initialise all pipe edges connections
        for i in range(0, len(graph_network[0])):
            for j in range(0, len(graph_network[1])):
                wn.add_pipe("pipe" + str(i + j), "node" + str(i), "node" + str(j), length=graph_network[i][j][0],
                            diameter=0.3048, roughness=100, minor_loss=0.0, status="OPEN")
        # setting time options for model
        wn.options.time.duration = 24 * 3600
        wn.options.time.hydraulic_timestep = 15 * 60
        wn.options.time.pattern_timestep = 15 * 60
        # for element in graph_network[0]:

        wn.write_inpfile('models/{}'.format(network_name))

        return wn

    def create_network_model_array(self, network_number, model_base_title):
        # Method to create large array of networks and save to external files for testing
        graph_network = self.create_network_graph()
        for i in range(0, network_number):
            self.create_network_model(graph_network, model_base_title + str(i) + ".inp")

    def create_network_model_data(self, network_title):
        # Method to take in created network models and simulate the necessary data and save in a seperate file
        model = Model(network_title)
        results = model.simulate_network()
        return results

    def create_network_model_data_array(self, model_base_title, data_number):
        results = []
        for i in range(0, data_number):
            results.append(self.create_network_model_data("models/" + model_base_title + str(i) + ".inp"))
        print(results[3])
        return results

    def create_leak_network_model(self):
        # TODO: implement leak model creation
        pass


class Model():
    def __init__(self, inp_file):
        self.title = inp_file
        self.wn = wntr.network.WaterNetworkModel(inp_file)
        self.get_graph()

    def simulate_network_EPANET(self):
        epanet_sim = wntr.sim.EpanetSimulator(self.wn)
        self.results = epanet_sim.run_sim()
        self.flowrate = self.results.link["flowrate"]
        self.pressure = self.results.node["pressure"]
        return self.results

    def create_dataset(self, days, nodes, location, start="1/1/2018"):
        for i in nodes:
            print(i)
            self.create_dataset_node(days, i, location, start=start)

        print("Finished creating data...")

    def create_dataset_node(self, days, node, location, start):
        import statistics
        dates = pd.date_range(start='1/1/2018', periods=97 * days, freq='15min')
        dates_str = dates.strftime("%Y-%m-%d %H:%M:%S").tolist()
        # print(dates)
        results = []
        results_pressure = []
        self.simulate_network_EPANET()
        result = self.get_pressure_at_node((node))
        std = statistics.stdev(result)
        print(std)

        for days in range(0, days):
          #  results.append(result)
					results.extend(result)
          if days %800:
          		results_pressure = list(lambda x: x+np.random.normal(0, std), results)
							print(results_pressure)
	        		df = pd.DataFrame({"DateTime": dates_str, "Pressure": results_pressure})
	        		if days > 800:
		        			with open(r'{}/{}.csv'.format(location, node), 'a') as f:
		    							df.to_csv(f, header=False)
	    				else:
		        			df.to_csv(r'{}/{}.csv'.format(location, node), index=True, header=True)
	        		results = []
            
        """    results_pressure = list(lambda x: x+np.random.normal(0, std), results)
        print(results_pressure)
        df = pd.DataFrame({"DateTime": dates_str, "Pressure": results_pressure})
        print(results[0])
       # for element in results:
        #    for item in element:
         #       results_pressure.append(item + np.random.normal(0, std))
        
        results_pressure = list(lambda x: x+np.random.normal(0, std), results)
        print(results_pressure)
        df = pd.DataFrame({"DateTime": dates_str, "Pressure": results_pressure})
        # sd = df.std()
        # results_pressure = []
        # for element in results:
        #    for item in element:
        #        results_pressure.append(item + 1.2 * np.random.normal(0, sd))
        # df = pd.DataFrame(results_pressure, index=dates, columns=["Pressure"])
        
"""
        print(df)

    def date(self, days):
        date = ""
        while days / 30 >= 1:
            days -= 30

    def date_time(self, seconds):
        days = 1
        hours = 0
        months = 1
        minutes = seconds / 60
        while (minutes / 60) >= 1:
            hours += 1
            minutes -= 60
            print(hours, minutes)
            if hours > 24:
                hours = 0
                days += 1
            if days > 30:
                months += 1

        print(months, days, hours, minutes)
        if minutes < 10 and hours < 10 and days < 10 and months < 10:
            time = "0{}/0{}/18 0{}:0{}:00".format(int(days), int(months), int(hours), int(minutes))
        elif hours < 10 and days < 10 and months < 10:
            time = "0{}/0{}/18 0{}:{}:00".format(int(days), int(months), int(hours), int(minutes))
        elif days < 10 and months < 10:
            time = "0{}/0{}/18 {}:{}:00".format(int(days), int(months), int(hours), int(minutes))
        elif months < 10:
            time = "0{}/{}/18 {}:{}:00".format(int(days), int(months), int(hours), int(minutes))
        else:
            time = "{}/{}/18 {}:{}:00".format(int(days), int(months), int(hours), int(minutes))
        return time

    def save_pressure(self, node):
        pressure_df = self.get_pressure_at_node(node)
        pressure_df.to_csv(r'/Users/Robin/Desktop/ML Project/Code/export_dataframe.csv', index=True, header=True)

    def simulate_network(self):
        wntr_sim = wntr.sim.WNTRSimulator(self.wn)
        self.results = wntr_sim.run_sim()
        self.flowrate = self.results.link["flowrate"]
        self.pressure = self.results.node["pressure"]
        return self.results

    def get_flowrate_at_link(self, link):
        flowrate_at_link = self.flowrate.loc[:, link]
        return flowrate_at_link

    def get_pressure_at_node(self, node):
        pressure_at_node = self.pressure.loc[:, node]
        return pressure_at_node

    def get_graph(self):
        self.graph = self.wn.get_graph()
        return self.graph

    def get_node(self, node):
        return self.graph[node]

    def save_results(self):
        pass

    def load_results(self):
        pass

    def save_model(self, model, filename):
        model.write_inpfile("models/{}".format(filename))

    def get_input_output_nodes(self, model):
        pipe_nodes = []
        pipes_iterator = model.pipes()
        junction_iterator = model.nodes()

        for item in pipes_iterator:
            pipe_nodes.append((str(item[1].start_node), str(item[1].end_node)))
            # TODO: figure out a resource efficient way to find the amount of nodes in a system
            junction_number = max()
        return pipe_nodes, junction_number

    def generate_random_leak_model(self, amount, models):
        for element in models:
            wn = wntr.network.WaterNetworkModel("models/" + element)
            pipe_nodes, junction_number = self.get_input_output_nodes(wn)
            leak_nodes = self.add_leaks(wn, random.randint(0, 15), junction_number)
            self.save_model(wn, element + "leak_model")

    def add_leaks(self, model, amount, junction_number):
        leak_nodes = []
        for i in range(0, amount):
            junction = "node" + str(random.randint(0, junction_number))
            model = wntr.morph.split_pipe(model, junction, junction + "_B", junction + "_leak_node")
            leak_node = model.get_node(junction + "_leak_node")
            leak_node.add_leak(model, area=random.uniform(0, 5.0), start_time=random.randint(0, 3600),
                               end_time=random.randint(3601, 24 * 3600))
            leak_nodes.append(leak_node)
        return leak_nodes

    def print_graph(self):
        graph = self.get_graph()
        node_degree = graph.degree()
        bet_cen = nx.betweenness_centrality(graph)

        wntr.graphics.plot_interactive_network(self.wn, node_range=[0, 500], auto_open=False)
        wntr.graphics.plot_network(self.wn, node_attribute=bet_cen, node_size=20,
                                   title='Betweenness Centrality')


def create_datasets(days, model):
    models = []
    for i in range(0, len(MEASUREMENT_NODES)):
        models.append(Model("models/DMA_04118v4x.INP"))
        models[i].days = days
    threads = []
    for i in range(0, len(MEASUREMENT_NODES)):
        thread = threading.Thread(target=models[i].create_dataset_node, args=(days, MEASUREMENT_NODES[i]))
        threads.append(thread)
        thread.start()
        time.sleep(3)
    for index, thread in enumerate(threads):
        thread.join()


if __name__ == "__main__":
    # model = Model("models/filename.inp")
    # print(model.get_node("node1"))
    # results = model.simulate_network()
    # print(model.get_pressure_at_node("node3"))
    # model.get_input_output_nodes(model.wn)
    # model.print_graph()
    # model.generate_random_leak_model(5, ["net1.inp"])
    # network_generator = NetworkGenerator(10, 0.5)
    # network_generator.initialise_graph()
    # graph_network = network_generator.create_network_graph()
    # network_generator.create_network_model(graph_network, "test1.inp")
    # network_generator.create_network_model_array(40, "model")
    # network_generator.create_network_model_data_array("model", 10)
    model = Model("DMA_03809v4x.INP")
    # results = model.simulate_network_EPANET()
    # print(model.get_pressure_at_node("40918044_28600378"))
    # model.save_pressure("40918044_28600378")
    model.create_dataset(80000, MEASUREMENT_NODES_2, location="DATASETS_SIM")

    # create_datasets(12, 1)
# model.print_graph()
