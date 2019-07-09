__author__ = "Robin M."

import wntr
import networkx as nx

# Model 1
wn = wntr.network.WaterNetworkModel()

# Adding pattertns
wn.add_pattern('pat1', [1])
wn.add_pattern('pat2', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# Adding Junctions
wn.add_junction('node1', base_demand=0.01, demand_pattern='pat1',
                elevation=50.0, coordinates=(1, 2))
wn.add_junction('node2', base_demand=0.02, demand_pattern='pat2',
                elevation=50.0, coordinates=(1, 3))
wn.add_junction('node3', base_demand=0.02, demand_pattern='pat1',
                elevation=30.0, coordinates=(1, 5))
wn.add_junction('node4', base_demand=0.02, demand_pattern='pat2',
                elevation=50.0, coordinates=(1, 6))
wn.add_junction('node5', base_demand=0.02, demand_pattern='pat2',
                elevation=50.0, coordinates=(2, 3))
wn.add_junction('node6', base_demand=0.02, demand_pattern='pat2',
                elevation=50.0, coordinates=(0, 3))
wn.add_junction('node7', base_demand=0.02, demand_pattern='pat2',
                elevation=50.0, coordinates=(5, 2))
wn.add_junction('node8', base_demand=0.02, demand_pattern='pat2',
                elevation=50.0, coordinates=(1, 0))
# Adding pump
wn.add_pump(name="pump1", start_node_name="node1", end_node_name="node2", speed=20, pump_parameter=150)
# Adding reservoir
wn.add_reservoir('res', base_head=125, head_pattern='pat1', coordinates=(0, 2))
# Adding pipes
wn.add_pipe('pipe1', 'node1', 'res', length=304.8, diameter=0.3048, roughness=100,
            minor_loss=0.0, status='OPEN')
wn.add_pipe('pipe2', 'node2', 'res', length=100, diameter=0.3048, roughness=100,
            minor_loss=0.0, status='OPEN')
wn.add_pipe('pipe3', 'node3', 'res', length=100, diameter=0.3048, roughness=100,
            minor_loss=0.0, status='OPEN')
wn.add_pipe('pipe4', 'node4', 'res', length=100, diameter=0.3048, roughness=100,
            minor_loss=0.0, status='OPEN')
wn.add_pipe('pipe5', 'node5', 'res', length=100, diameter=0.3048, roughness=100,
            minor_loss=0.0, status='OPEN')
wn.add_pipe('pipe6', 'node6', 'res', length=100, diameter=0.3048, roughness=100,
            minor_loss=0.0, status='OPEN')
wn.add_pipe('pipe7', 'node7', 'res', length=100, diameter=0.3048, roughness=100,
            minor_loss=0.0, status='OPEN')
wn.add_pipe('pipe8', 'node8', 'res', length=100, diameter=0.3048, roughness=100,
            minor_loss=0.0, status='OPEN')
wn.add_pipe('pipe9', 'node1', 'node2', length=100, diameter=0.3048, roughness=100,
            minor_loss=0.0, status='OPEN')
# Simulation time options
wn.options.time.duration = 24 * 3600
wn.options.time.hydraulic_timestep = 15 * 60
wn.options.time.pattern_timestep = 60 * 60

wn.write_inpfile('models/filename.inp')
