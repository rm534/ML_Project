import pandas as pd
import model
import numpy as np


def read_csv(filename, data_len):
    df = pd.read_csv(filename, nrows=data_len)
    return df

def read_data_list(DMA):
    sim_dfs = []
    real_dfs = []
    if DMA == "03809":
        nodes = model.MEASUREMENT_NODES_2
    else:
        nodes = model.MEASUREMENT_NODES
    for element in nodes:
        sim_dfs.append(pd.read_csv("DATASETS_REAL/03809_sim/{}.csv".format(element), names=["DateTime","Pressure"]))
        real_dfs.append(pd.read_csv("DATASETS_REAL/03809_real/{}.csv".format(element), names=["DateTime","Pressure"]))
    #print(sim_dfs[2], sim_dfs[1])
    return sim_dfs, real_dfs

# TODO: chagne to read_dataset
def read_data(data_len):
    data_read = []
    dataset_list = []
    dataset_list_total = []
    data_size = data_len
    for element in model.MEASUREMENT_NODES_2:
        if model.MEASUREMENT_NODES_2.index(element) == 0:
            data_read.append(read_time_data('/Users/Robin/Desktop/ML Project/Code/DATASETS_SIM/{}.csv'.format(element), data_len))
        data_read.append(
            read_csv('/Users/Robin/Desktop/ML Project/Code/DATASETS_SIM/{}.csv'.format(element), data_len)[
                "Pressure"].to_list())

    for i in range(0, data_size):
        for element in data_read:
            dataset_list.append(element[i])
        dataset_list_total.append(dataset_list)
        dataset_list = []

    dataset = np.array(dataset_list_total)
    #print(dataset.shape)
    return dataset
    
def compose_data_comparison(location, DMA):
	# access simulated data frames and read dataframes as lists
	sim_dfs, real_dfs = read_data_list(DMA)
		for i in range (0, len(MEASUREMENT_NODES_2)):
			# find first and last timestamp from real_dfs
			# TODO: timestamp of real data might vary from simulated data, verify this...
			start = real_dfs[i]["DateTime"].to_list()[0]
			end = real_dfs[i]["DateTime"].to_list()[-1]
			# find index of sim DateTime for starting value of real
			DateTime = sim_dfs[i]["DateTime"].to_list()
			start_ind = DateTime.index(start)
			end_ind = DateTime.index(end)
			DateTime = DateTime[start_ind:end_ind]
			pressure_real = real_dfs[i]["Pressure"].to_list()
			pressure_sim = sim_dfs[i]["Pressure"].to_list()[start_ind:end_ind]
			diff = []
			for j in range(0,len(DateTime)):
					diff.append(pressure_real[j]-pressure_sim[j])
			
			# create a dataframe that incorporates all the data real and simulated for the same timeframe
			var_dfs = pd.DataFrame({"DateTime":DateTime, "Pressure_r":pressure_real, "Pressure_s":pressure_sim, "difff":diff})
			# Save to disk
			var_dfs.to_csv(r'{}/{}_difference.csv'.format(location, model.MEASUREMENT_NODES_2[i]), index=True, header=True)

def compose_data_comparison(location, DMA):
	# access simulated data frames and read dataframes as lists
	sim_dfs, real_dfs = read_data_list(DMA)
		for i in range (0, len(MEASUREMENT_NODES_2)):
			# find first and last timestamp from real_dfs
			# TODO: timestamp of real data might vary from simulated data, verify this...
			start = real_dfs[i]["DateTime"].to_list()[0]
			end = real_dfs[i]["DateTime"].to_list()[-1]
			# find index of sim DateTime for starting value of real
			DateTime = sim_dfs[i]["DateTime"].to_list()
			start_ind = DateTime.index(start)
			end_ind = DateTime.index(end)
			DateTime = DateTime[start_ind:end_ind]
			pressure_real = real_dfs[i]["Pressure"].to_list()
			pressure_sim = sim_dfs[i]["Pressure"].to_list()[start_ind:end_ind]
			diff = []
			for j in range(0,len(DateTime)):
					diff.append(pressure_real[j]-pressure_sim[j])
			
			# create a dataframe that incorporates all the data real and simulated for the same timeframe
			var_dfs = pd.DataFrame({"DateTime":DateTime, "Pressure_r":pressure_real, "Pressure_s":pressure_sim, "difff":diff})
			# Save to disk
			var_dfs.to_csv(r'{}/{}_difference.csv'.format(location, model.MEASUREMENT_NODES_2[i]), index=True, header=True)
		
		

def compose_data_real(data_len, location, DMA):
    # To compose real data, simulated data is added to missing values in all other datasets
 
    sim_dfs, real_dfs = read_data_list(DMA)
    assert len(sim_dfs) == len(real_dfs)
    for i in range(0, len(sim_dfs)):
        print("i:{}".format(i))
        date_time_final=[]
        data_real_final_list = []
        data_real_final = []
        sim_dfs[i] = sim_dfs[i].drop(0)
        real_dfs[i] = real_dfs[i].drop(0)
        date_time = sim_dfs[i]["DateTime"].to_list()
        date_time_real = real_dfs[i]["DateTime"].to_list()
        date_time_real = assert_date_time(date_time_real)
        date_time_real = date_time_real_into_pd(date_time_real)
        t=0
        for j in range(0, len(date_time)):
            #print(date_time_real[t])
            #print(date_time[j])
            if date_time[j] != date_time_real[t]:
                date_time_final.append(date_time[j])
                data_real_final_list.append(sim_dfs[i].iloc[j]["Pressure"])
            elif date_time[j] == date_time_real[t]:
                print("same!")
                t+=1
                data_real_final_list.append(real_dfs[i].iloc[j]["Pressure"])
                date_time_final.append(date_time[j])

        print(len(data_real_final_list))
        df = pd.DataFrame({"DateTime": date_time_final, "Pressure": data_real_final_list})
        df.to_csv(r'{}/{}.csv'.format(location, model.MEASUREMENT_NODES_2[i]), index=True, header=True)
        
        
def read_data_real(data_len):
    data_read = []
    dataset_list = []
    dataset_list_total = []
    data_size = data_len
    for element in model.MEASUREMENT_NODES_2:
        if model.MEASUREMENT_NODES_2.index(element) == 0:
            data_read.append(read_time_data('/Users/Robin/Desktop/ML Project/Code/DATASETS_REAL/03809_composition/{}.csv'.format(element), data_len))
        data_read.append(
            read_csv('/Users/Robin/Desktop/ML Project/Code/DATASETS_REAL/03809_composition/{}.csv'.format(element), data_len)[
                "Pressure"].to_list())

    for i in range(0, data_size):
        for element in data_read:
            dataset_list.append(element[i])
        dataset_list_total.append(dataset_list)
        dataset_list = []

    dataset = np.array(dataset_list_total)
    print(dataset.shape)
    return dataset

def read_time_data(filename, data_len):
    df = read_csv(filename, data_len=data_len)
    time_date = df["DateTime"].to_list()
    time_date_final = []
    for element in time_date:
        time_date_final.append(normalise_time_from_ts(element))
    return time_date_final


def normalise_time_from_ts(DateTime):
    time = DateTime[11:]
    #print("time: {}".format(time))
    hours = int(time[:2])
    #print("hours: {}".format(hours) )
    minutes = int(time[3:5])
    time_norm = ((hours*60)+minutes)/1440
    return time_norm


def assert_date_time(date_time):
    for item in date_time:
        if len(item) != 19:
            date_time[date_time.index(item)] = item + ":00"
    return date_time

def date_time_real_into_pd(date_time):
    #print(date_time)
    date_time_new = pd.to_datetime(date_time, format="%d/%m/%Y %H:%M:%S")
    #print(type(date_time_new))
    return date_time_new.strftime("%Y-%m-%d %H:%M:%S").tolist()
# print(real_dfs[i])


if __name__ == "__main__":
    compose_data_real(data_len=10, DMA="03809", location="DATASETS_REAL/03809_composition")
    #print(read_time_data("DATASETS_REAL/03809_composition/39805881_29979691.csv", 100))
    read_data_real(10)
