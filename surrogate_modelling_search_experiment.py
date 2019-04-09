# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:53:16 2019

@author: James
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 7 12:49:41 2019

@author: James
"""
from smse_experiment_functions import simulation_ga, batch_simulation, train_regression_surrogate_model, train_classifier_surrogate, surrogate_ga, surrogate_ga_sim_eval
import os
import sys
import random
import datetime

######################################## GLOBALS ###################################
wd = os.getcwd()
MU = 100
LAMBDA = 25
GA_GENERATIONS = 196
SIMGA_RUNS = 10
SURRGA_RUNS = 100
COMPAREGA_RUNS = 10
SIMGA_MAXTIME = 10000
BATCH_MAXTIME = 10000
SM_MAXTIME = 6
SURRGA_MAXTIME = 100
COMPAREGA_MAXTIME = 10000
MAX_BATCH = 50000
SM_DATASIZES = [5000,10000,15000,25000,35000,50000]
#SM_DATASIZES = [50000]
SM_TRAINING = [0.05, 0.1, 0.25, 0.5, 0.75]


#Set in progress experimental directory and initilise seeds and time recording csv
BASE_DIRECTORY = wd+"/"
SIMGA = BASE_DIRECTORY+"simulation_ga/"
BATCHSIM = BASE_DIRECTORY+"batch_simulation_data/"
SURR = BASE_DIRECTORY+"surrogate_model_training/"
SURRGA = BASE_DIRECTORY+"surrogate_ga/"
SURRSIM = BASE_DIRECTORY+"surrogate_vs_sim_comparison/"
TEST = BASE_DIRECTORY+"testing/"
directories = [BASE_DIRECTORY,SIMGA,BATCHSIM,SURR,SURRGA,SURRSIM,TEST]
for d in directories:
    if not os.path.exists(d):
        os.mkdir(d)
    if not os.path.exists(d+"seeds.csv"):
        open(d+"seeds.csv","w").close()
    if not os.path.exists(d+"times.csv"):
        open(d+"times.csv","w").close()


######################################## SIMULATION GA BASELINE EXPERIMENT ##############################################
#Preliminary experiment, simulation ga performance averages
#GA limitation discovery/performance in terms of average and variation graph. Should show tuning towards high quality  
def perform_x_continuous_simulation_ga(x,mu,lam,gagen,gat=9999):
    print("Baseline experiment: Simulation GAs (Continuous primary fitness implementation) to ascertain average performance and variance.")
    loc = SIMGA+"/continuous_fitness/"
    if not os.path.exists(loc):
        os.mkdir(loc)
    #Set and record seed
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    sf = open(BASE_DIRECTORY+"seeds.csv","a")
    sf.write("simulation_ga_cont_fitness,"+str(seed)+"\n")
    sf.close()
    sf2 = open(SIMGA+"seeds.csv","a")
    sf2.write("simulation_ga_cont_fitness,"+str(seed)+"\n")
    sf2.close()
    #Track experiment time
    initial_time = datetime.datetime.now()
    #Run experiment over x sim GAs
    for i in range(x):
        simulation_ga(mu,lam,gagen,gat,loc,"continuous")
        ################################################################################################pred_prey_ga_random(mu,lam,gagen,gat,"/continuous_fitness/",seed)
        print("completed continuous sim GA ",i+1," out of ",x)
        final_time = datetime.datetime.now()
        simga_time_taken = final_time-initial_time
        minutes = int(simga_time_taken.seconds/60)
        if minutes>gat:
            print("Time limit reached, exiting with ",i," completed GA")
            break
    #Log time
    tf = open(BASE_DIRECTORY+"times.csv","a")
    tf.write(str(x)+",simulation_ga_cont_fitness_total_time,"+str(simga_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    tf = open(SIMGA+"times.csv","a")
    tf.write(str(x)+",simulation_ga_cont_fitness_total_time,"+str(simga_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    print("Continuous sim GA performance experiment complete.\n")
    return

def perform_x_discrete_simulation_ga(x,mu,lam,gagen,gat=9999):
    print("Baseline experiment: Simulation GAs (Discrete primary fitness implementation) to ascertain average performance and variance.")
    loc = SIMGA+"/discrete_fitness/"
    if not os.path.exists(loc):
        os.mkdir(loc)
    #Set and record seed
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    sf = open(BASE_DIRECTORY+"seeds.csv","a")
    sf.write("simulation_ga_discrete_fitness,"+str(seed)+"\n")
    sf.close()
    sf2 = open(SIMGA+"seeds.csv","a")
    sf2.write("simulation_ga_discrete_fitness,"+str(seed)+"\n")
    sf2.close()
    #Track experiment time
    initial_time = datetime.datetime.now()
    #Run experiment over x sim GAs
    for i in range(x):
        simulation_ga(mu,lam,gagen,gat,loc,"discrete")
        ################################################################################################pred_prey_ga_random(mu,lam,gagen,gat,"/continuous_fitness/",seed)
        print("completed discrete sim GA ",i+1," out of ",x)
        final_time = datetime.datetime.now()
        simga_time_taken = final_time-initial_time
        minutes = int(simga_time_taken.seconds/60)
        if minutes>gat:
            print("Time limit reached, exiting with ",i," completed GA")
            break
    #Log time
    tf = open(BASE_DIRECTORY+"times.csv","a")
    tf.write(str(x)+",simulation_ga_discrete_fitness_total_time,"+str(simga_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    tf = open(SIMGA+"times.csv","a")
    tf.write(str(x)+",simulation_ga_discrete_fitness_total_time,"+str(simga_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    print("Discrete sim GA performance experiment complete.\n")
    return


########################################  BATCH SIMULATION SURROGATE TRAINING DATA GENERATION ################################################
#   Batch simulate the desired number of paramter vectors
def x_batch_simulations(x, typ, time=9999):
    print("Experiment: Perform required number of simulations in batch to generate training data for surrogate model.")
    #Set and record seed
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    sf = open(BASE_DIRECTORY+"seeds.csv","a")
    sf.write(typ+"_batch_sim_data_generation,"+str(seed)+"\n")
    sf.close()
    sf2 = open(BATCHSIM+"seeds.csv","a")
    sf2.write(typ+"_batch_sim_data_generation,"+str(seed)+"\n")
    sf2.close()
    #Track experiment time
    initial_time = datetime.datetime.now()
    #Run x batch simulations by queue with maximum runtime time
    batch_simulation(x,time,typ,seed,BATCHSIM)
    final_time = datetime.datetime.now()
    batch_data_time_taken = final_time-initial_time
    #Log time
    tf = open(BASE_DIRECTORY+"times.csv","a")
    tf.write(str(x)+","+typ+"_batch_sim_data_generation,"+str(batch_data_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    tf = open(BATCHSIM+"times.csv","a")
    tf.write(str(x)+","+typ+"_batch_sim_data_generation,"+str(batch_data_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    print("Completed training data batch simulations.\n")
    return


############################################# TRAIN SURROGATE MODELS ###############################################
#   Train surrogate models for as long as time permits. Log and save the best performing models
def train_regression_surrogate_models(data,training,batch_data_loc,loc,time=9999):
    print("Experiment: Train surrogate models until time limit is reached, save best performing models.")
    #Set and record seed
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    sf = open(BASE_DIRECTORY+"seeds.csv","a")
    sf.write("regression_surrogate_training,"+str(seed)+"\n")
    sf.close()
    sf2 = open(SURR+"seeds.csv","a")
    sf2.write("regression_surrogate_training,"+str(seed)+"\n")
    sf2.close()
    #Track experiment time
    initial_time = datetime.datetime.now()
    #Train regression models with maximum runtime time for each type of model to be trained, return the number of models trained
    x, completion_time1 = train_regression_surrogate_model(data,training,initial_time,time,batch_data_loc,loc,1)
    y, completion_time2 = train_regression_surrogate_model(data,training,completion_time1,time,batch_data_loc,loc,2)
    z, final_time = train_regression_surrogate_model(data,training,completion_time2,time,batch_data_loc,loc,3)
    #Log time
    tf = open(BASE_DIRECTORY+"times.csv","a")
    tf.write(str(x)+",regression_surrogate_training_primary,"+str(completion_time1-initial_time)+",start,"+str(initial_time)+",finish,"+str(completion_time1)+"\n")
    tf.write(str(y)+",regression_surrogate_training_secondary,"+str(completion_time2-completion_time1)+",start,"+str(completion_time1)+",finish,"+str(completion_time2)+"\n")
    tf.write(str(z)+",regression_surrogate_training_tertiary,"+str(final_time-completion_time2)+",start,"+str(completion_time2)+",finish,"+str(final_time)+"\n")
    tf.close()
    tf = open(SURR+"times.csv","a")
    tf.write(str(x)+",regression_surrogate_training_primary,"+str(completion_time1-initial_time)+",start,"+str(initial_time)+",finish,"+str(completion_time1)+"\n")
    tf.write(str(y)+",regression_surrogate_training_secondary,"+str(completion_time2-completion_time1)+",start,"+str(completion_time1)+",finish,"+str(completion_time2)+"\n")
    tf.write(str(z)+",regression_surrogate_training_tertiary,"+str(final_time-completion_time2)+",start,"+str(completion_time2)+",finish,"+str(final_time)+"\n")
    tf.close()
    print("Completed training regression surrogate models.\n")
    return

def train_classifier_surrogate_model(data,training,batch_data_loc,loc,time=9999):
    print("Experiment: Train surrogate models until time limit is reached, save best performing model.")
    #Set and record seed
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    sf = open(BASE_DIRECTORY+"seeds.csv","a")
    sf.write("classifier_surrogate_training,"+str(seed)+"\n")
    sf.close()
    sf2 = open(SURR+"seeds.csv","a")
    sf2.write("classifier_surrogate_training,"+str(seed)+"\n")
    sf2.close()
    #Track experiment time
    initial_time = datetime.datetime.now()
    #Train classifier model with maximum runtime time, return the number of models trained
    x, final_time = train_classifier_surrogate(data,training,initial_time,time,batch_data_loc,loc)
    surrogate_training_time_taken = final_time-initial_time
    #Log time
    tf = open(BASE_DIRECTORY+"times.csv","a")
    tf.write(str(x)+",classifier_surrogate_training,"+str(surrogate_training_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    tf = open(SURR+"times.csv","a")
    tf.write(str(x)+",classifier_surrogate_training,"+str(surrogate_training_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    print("Completed training classifier surrogate models.\n")
    return


############################################ SURROGATE GUIDED GA ###############################################
# Perform multiple surrogate guided GA to measure average fitness and variance
def perform_x_continuous_surrogate_ga(models,x,mu,lam,gagen,gat=9999):
    loc = SURRGA+"/continuous_fitness/"
    if not os.path.exists(loc):
        os.mkdir(loc)
    print("Surrogate GAs (Continuous primary fitness implementation) to ascertain average performance and variance.")
    #Set and record seed
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    sf = open(BASE_DIRECTORY+"seeds.csv","a")
    sf.write("surrogate_ga_cont_fitness,"+str(seed)+"\n")
    sf.close()
    sf2 = open(SURRGA+"seeds.csv","a")
    sf2.write("surrogate_ga_cont_fitness,"+str(seed)+"\n")
    sf2.close()
    #Track experiment time
    initial_time = datetime.datetime.now()
    #Run experiment over x surr GAs
    final_time = surrogate_ga(x,models,mu,lam,gagen,gat,SURR,loc,"continuous")
    surrga_time_taken = final_time-initial_time
    #Log time
    tf = open(BASE_DIRECTORY+"times.csv","a")
    tf.write(str(x)+",surrogate_ga_cont_fitness_total_time,"+str(surrga_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    tf = open(SURRGA+"times.csv","a")
    tf.write(str(x)+",surrogate_ga_cont_fitness_total_time,"+str(surrga_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    print("Continuous surr GA performance experiment complete.\n")
    return

def perform_x_discrete_surrogate_ga(models,x,mu,lam,gagen,gat=9999):
    print("Surrogate GAs (Discrete primary fitness implementation) to ascertain average performance and variance.")
    loc = SURRGA+"/discrete_fitness/"
    if not os.path.exists(loc):
        os.mkdir(loc)
    #Set and record seed
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    sf = open(BASE_DIRECTORY+"seeds.csv","a")
    sf.write("surrogate_ga_discrete_fitness,"+str(seed)+"\n")
    sf.close()
    sf2 = open(SURRGA+"seeds.csv","a")
    sf2.write("surrogate_ga_discrete_fitness,"+str(seed)+"\n")
    sf2.close()
    #Track experiment time
    initial_time = datetime.datetime.now()
    #Run experiment over x surr GAs
    final_time = surrogate_ga(x,models,mu,lam,gagen,gat,SURR,loc,"discrete")
    surrga_time_taken = final_time-initial_time
    #Log time
    tf = open(BASE_DIRECTORY+"times.csv","a")
    tf.write(str(x)+",surrogate_ga_discrete_fitness_total_time,"+str(surrga_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    tf = open(SURRGA+"times.csv","a")
    tf.write(str(x)+",surrogate_ga_discrete_fitness_total_time,"+str(surrga_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    print("Discrete surr GA performance experiment complete.\n")
    return


############################################# SURROGATE PREDICTED VS SIMULATION EVALUATED FITNESSES OF GA POP #########################
# Perform multiple surrogate guided GA while evaluating the true fitness of the discovered population by simulation (displays surrogate prediction accuracy/true quality of solutions)
def perform_x_continuous_compare_ga(models,x,mu,lam,gagen,gat=9999):
    print("Surrogate GAs (Continuous primary fitness implementation) with discovered populations evalauted by simulation at each generation.")
    loc = TEST+"/continuous_fitness/"
    if not os.path.exists(loc):
        os.mkdir(loc)
    #Set and record seed
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    sf = open(BASE_DIRECTORY+"seeds.csv","a")
    sf.write("comparison_ga_cont,"+str(seed)+"\n")
    sf.close()
    sf2 = open(SURRSIM+"seeds.csv","a")
    sf2.write("comparison_ga_cont,"+str(seed)+"\n")
    sf2.close()
    #Track experiment time
    initial_time = datetime.datetime.now()
    #Run experiment over x surr GAs with pop evaluated by sim each generation
    final_time = surrogate_ga_sim_eval(x,models,mu,lam,gagen,gat,loc,SURR,"continuous")
    comparison_ga_time_taken = final_time-initial_time
    #Log time
    tf = open(BASE_DIRECTORY+"times.csv","a")
    tf.write(str(x)+",comparison_ga_cont_total_time,"+str(comparison_ga_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    tf = open(SURRSIM+"times.csv","a")
    tf.write(str(x)+",comparison_ga_cont_total_time,"+str(comparison_ga_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    print("Continuous comparison GA experiment complete.\n")
    return

def perform_x_discrete_compare_ga(models,x,mu,lam,gagen,gat=9999):
    print("Surrogate GAs (Discrete primary fitness implementation) with discovered populations evalauted by simulation at each generation.")
    loc = TEST+"/discrete_fitness/"
    if not os.path.exists(loc):
        os.mkdir(loc)
    #Set and record seed
    seed = random.randrange(sys.maxsize)
    random.seed(seed)
    sf = open(BASE_DIRECTORY+"seeds.csv","a")
    sf.write("comparison_ga_discrete,"+str(seed)+"\n")
    sf.close()
    sf2 = open(SURRSIM+"seeds.csv","a")
    sf2.write("comparison_ga_discrete,"+str(seed)+"\n")
    sf2.close()
    #Track experiment time
    initial_time = datetime.datetime.now()
    #Run experiment over x surr GAs with pop evaluated by sim each generation
    final_time = surrogate_ga_sim_eval(x,models,mu,lam,gagen,gat,loc,SURR,"discrete")
    comparison_ga_time_taken = final_time-initial_time
    #Log time
    tf = open(BASE_DIRECTORY+"times.csv","a")
    tf.write(str(x)+",comparison_ga_discrete_total_time,"+str(comparison_ga_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    tf = open(SURRSIM+"times.csv","a")
    tf.write(str(x)+",comparison_ga_discrete_total_time,"+str(comparison_ga_time_taken)+",start,"+str(initial_time)+",finish,"+str(final_time)+"\n")
    tf.close()
    print("Discrete comparison GA experiment complete.\n")
    return


######################################### TEST EXPERIMENTS ##################################################
#perform_x_continuous_simulation_ga(2,2,1,3,5)#WORKS
#perform_x_discrete_simulation_ga(2,2,1,3,5)#WORKS
#x_batch_simulations(500,"continuous",1)#WORKS
#x_batch_simulations(500,"discrete",1)#WORKS
#train_regression_surrogate_models([100,250,500],[0.5,0.75,0.95],BATCHSIM+"/continuous/",SURR,2)#WORKS
#train_classifier_surrogate_model([100,250,500],[0.5,0.75,0.95],BATCHSIM+"/discrete/",SURR,2)#WORKS
#perform_x_continuous_surrogate_ga([100,250,500],250,2,1,3,1)#WORKS
#perform_x_discrete_surrogate_ga([100,250,500],250,2,1,3,1)#WORKS
#perform_x_continuous_compare_ga([100,250,500],2,2,1,3,2)#WORKS
#perform_x_discrete_compare_ga([100,250,500],2,2,1,3,2)#WORKS
    

######################################### MEDIUM SCALE TEST #################################################
#perform_x_continuous_simulation_ga(5,20,5,16,100)
#perform_x_discrete_simulation_ga(5,20,5,16,100)
#x_batch_simulations(500,"continuous",100)
#x_batch_simulations(500,"discrete",100)
#train_regression_surrogate_models([250,400,500],[0.5,0.75,0.95],BATCHSIM+"/continuous/",SURR,5)
#train_classifier_surrogate_model([250,400,500],[0.5,0.75,0.95],BATCHSIM+"/discrete/",SURR,5)
#perform_x_continuous_surrogate_ga([250,400,500],1000,20,5,16,100)
#perform_x_discrete_surrogate_ga([250,400,500],1000,20,5,16,100)
#perform_x_continuous_compare_ga([250,400,500],2,20,5,16,100)
#perform_x_discrete_compare_ga([250,400,500],2,20,5,16,100)

######################################### RUN EXPERIMENTS ###################################################

###### SIMULATION GA #####
perform_x_continuous_simulation_ga(SIMGA_RUNS,MU,LAMBDA,GA_GENERATIONS,SIMGA_MAXTIME)
perform_x_discrete_simulation_ga(SIMGA_RUNS,MU,LAMBDA,GA_GENERATIONS,SIMGA_MAXTIME)
#
#
####### BATCH SIM FOR SURROGATE DATA #####
x_batch_simulations(MAX_BATCH,"continuous",BATCH_MAXTIME)
x_batch_simulations(MAX_BATCH,"discrete",BATCH_MAXTIME)
#
#
###### TRAIN SURROGATE MODELS #####
train_regression_surrogate_models(SM_DATASIZES,SM_TRAINING,BATCHSIM+"continuous/",SURR,SM_MAXTIME)
train_classifier_surrogate_model(SM_DATASIZES,SM_TRAINING,BATCHSIM+"discrete/",SURR,SM_MAXTIME)
#        
#
###### SURROGATE GAs #####
perform_x_continuous_surrogate_ga(SM_DATASIZES,SURRGA_RUNS,MU,LAMBDA,GA_GENERATIONS,SURRGA_MAXTIME)
perform_x_discrete_surrogate_ga(SM_DATASIZES,SURRGA_RUNS,MU,LAMBDA,GA_GENERATIONS,SURRGA_MAXTIME)
#        
#
###### SURR VS SIM FITNESS COMPARISON #####
perform_x_continuous_compare_ga(SM_DATASIZES,COMPAREGA_RUNS,MU,LAMBDA,GA_GENERATIONS,COMPAREGA_MAXTIME)
perform_x_discrete_compare_ga(SM_DATASIZES,COMPAREGA_RUNS,MU,LAMBDA,GA_GENERATIONS,COMPAREGA_MAXTIME)
