# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:53:53 2019

@author: James
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 7 13:12:53 2019

@author: James
"""

from deap import base
from deap import tools
from deap import creator
import os
import sys
import random
import numpy as np
import datetime
import queue
import threading
import pycuda.driver as cuda
import pycuda.autoinit
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
try:
    import cPickle as pickle
except:
    import pickle

###############################GLOBALS#########################
parameter_limits = [[0,5000],[0,5000],[0,5000],[0.0,0.25],[0.0,0.25],[0,200],[0,200],[0,200]]
#Maximum amount of parameter mutation
MUTATION = 0.25
GPUS_AVAILABLE = cuda.Device(0).count()
pop_queue = None
pop_queue_lock = threading.Lock()
exit_pop_queue = 0
batch_queue = None
batch_queue_lock = threading.Lock()
exit_batch_queue = 0
batch_times = [0]*GPUS_AVAILABLE

##############################SIMGA#############################
def simulation_ga(m,l,g,time,loc,c,s=1):
    global curr_pop, gen, s1, s2, s3, s4, population, toolbox, logbook
    if not os.path.exists(loc+"runs/"):
        os.mkdir(loc+"runs/")
    runloc = loc+"runs/"
    curr_pop = 0
    #Initialise GA tools
    creator.create("Fitness", base.Fitness, weights=(1.0,))
    if c=="continuous":
        creator.create("Fitnesses", base.Fitness, weights=(0.001,0.01,-0.00001))
    else:
        creator.create("Fitnesses", base.Fitness, weights=(1.0,0.01,-0.00001))
    creator.create("Individual", list, fitness=creator.Fitness, fitnesses=creator.Fitnesses)
    toolbox = base.Toolbox()
    toolbox.register("select_parents", select_parents, toolbox)
    toolbox.register("mutate", mutate, toolbox)
    toolbox.register("mate", mate, 0.5)
    toolbox.register("individual", initialise_individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #Initialise Logging tools
    s1 = tools.Statistics(lambda ind: ind.fitness.values[0])
    s1.register("avg", np.mean)
    s1.register("std", np.std)
    s1.register("min", np.min)
    s1.register("max", np.max)
    s2 = tools.Statistics(lambda ind: ind.fitnesses.values[0])
    s2.register("p-avg", np.mean)
    s2.register("p-std", np.std)
    s2.register("p-min", np.min)
    s2.register("p-max", np.max)
    s3 = tools.Statistics(lambda ind: ind.fitnesses.values[1])
    s3.register("s-avg", np.mean)
    s3.register("s-std", np.std)
    s3.register("s-min", np.min)
    s3.register("s-max", np.max)
    s4 = tools.Statistics(lambda ind: ind.fitnesses.values[2])
    s4.register("t-avg", np.mean)
    s4.register("t-std", np.std)
    s4.register("t-min", np.min)
    s4.register("t-max", np.max)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (s1.fields+s2.fields+s3.fields+s4.fields if s1 and s2 and s3 and s4 else [])
    gen = 0
    #Initialise random population of MU individuals
    population = toolbox.population(n=m)
    #Evaluate and log population fitnesses
    eval_count = len(population)
    start_time = datetime.datetime.now()
    print("Initial population evalauation (Generation 0)")
    initial_fitnesses = simulation_evaluate_population(population,loc,c)
    unique_run_seed = random.randrange(sys.maxsize)
    pr = runloc+str(unique_run_seed)+".csv"
    if not os.path.exists(pr):
        population_record = open(pr,"w")
    else:
        population_record = open(runloc+str(unique_run_seed)+"(1).csv","w")
    population_record.write("generation,0,mu,"+str(m)+",lambda,"+str(l)+"\n")
    for i in range(len(initial_fitnesses)):
        population[i].fitness.values = initial_fitnesses[i][0]
        population[i].fitnesses.values = initial_fitnesses[i][1]
        population_record.write("\t")
        for j in population[i][0].tolist():
            population_record.write(str(j)+",")
        population_record.write("fitness,"+str(population[i].fitness.values)+",fitnesses,"+str(population[i].fitnesses.values)+"\n")
    population_record.close()   
    log(logbook, population, gen, len(population))
    #Perform GA
    end_time,time_taken, end_pop = pred_prey_ga_sim(m,l,g,time,start_time,eval_count,c,pr,loc,runloc,unique_run_seed)
    #Log results and time taken
    if c=="continuous":
        if os.path.exists(loc+"continuous_sim_ga_performances.csv"):
            pf = open(loc+"continuous_sim_ga_performances.csv","a")
        else:
            pf = open(loc+"continuous_sim_ga_performances.csv","w")
        pf.write(str(logbook)+"\n")
        pf.close()
        if not os.path.exists(loc+"times.csv"):
            open(loc+"times.csv","w").close()
        time = open(loc+"times.csv","a")
        time.write("continuous_simulation_ga,"+str(unique_run_seed)+",start,"+str(start_time)+",end,"+str(end_time)+",total,"+str(time_taken)+"\n")
        time.close()
    else:
        if os.path.exists(loc+"discrete_sim_ga_performances.csv"):
            pf = open(loc+"discrete_sim_ga_performances.csv","a")
        else:
            pf = open(loc+"discrete_sim_ga_performances.csv","w")
        pf.write(str(logbook)+"\n")
        pf.close()
        if not os.path.exists(loc+"times.csv"):
            open(loc+"times.csv","w").close()
        time = open(loc+"times.csv","a")
        time.write("discrete_simulation_ga,"+str(unique_run_seed)+",start,"+str(start_time)+",end,"+str(end_time)+",total,"+str(time_taken)+"\n")
        time.close()
    return

def pred_prey_ga_sim(m,l,g,time,start_time,eval_count,ft,pr,loc,runloc,seed):
    global curr_pop, toolbox, population, logbook, gen
    gen = 0
    conditions_met = 0
    optimals = []
    ocount = 0
    #While all stop conditions have yet to be reached, perform generation
    while(gen<g and conditions_met==0):
        gen += 1
        print("\t Generation:",gen)
        nevals = 0
        curr_pop = 0
        #Generate offspring
        offspring = []
        evaluations = []
        #crossover is being used, it is done before mutation
        for i in range(l):
            new = random.uniform(0,1)
            if new<0.5:
                p1 = toolbox.individual()
            else:
                p1, p2 = [toolbox.clone(x) for x in toolbox.select_parents(population, 2)]
                toolbox.mate(p1, p2)
            offspring += [p1]
        #Mutate
        for off in offspring:
            off, = toolbox.mutate(off)
        nevals += len(offspring)
        evaluations = simulation_evaluate_population(offspring,loc,ft)
        for i in range(len(evaluations)):
            offspring[i].fitness.values = evaluations[i][0]
            offspring[i].fitnesses.values = evaluations[i][1]
        eval_count += nevals
        # Select the next generation, favouring the offspring in the event of equal fitness values
        population, new_individuals = favour_offspring(population, offspring, m)        
        #Print a report about the current generation
        if nevals > 0:
            log(logbook, population, gen, nevals)
        #Save to file in case of early exit
        log_fitness = open(loc+"fitness_log.csv","w")
        log_fitness.write(str(logbook)+"\n")
        log_fitness.close()
        if os.path.exists(pr):
            population_record = open(pr,"a")
        else:
            population_record = open(runloc+seed+"(1).csv","a")
        check_nonunique = []
        for p in population:
            population_record.write("\t")
            for q in p[0].tolist():
                population_record.write(str(q)+",")
            population_record.write("fitness,"+str(p.fitness.values)+",fitnesses,"+str(p.fitnesses.values)+"\n")
            if ft==1 and p.fitness.values[0]>0.95 and p.fitnesses.values[0]>900 and p.fitnesses.values[2]<15000:
                for opt in optimals:
                    check_nonunique.append(all(elem in p[0][:-1] for elem in opt[0][:-1]))
                if not any(check_nonunique):
                    optimals.append((p,gen))
            if ft==0 and p.fitness.values[0]>0.95 and p.fitnesses.values[0]>0 and p.fitnesses.values[2]<15000:
                for opt in optimals:
                    check_nonunique.append(all(elem in p[0][:-1] for elem in opt[0][:-1]))
                if not any(check_nonunique):
                    optimals.append((p,gen))
        population_record.write("Simulation,generation,"+str(gen)+"\n")
        population_record.close()
        end_time = datetime.datetime.now()
        time_taken = end_time-start_time
        opti = optimals[ocount:]
        if len(opti)>0:
            opt = open(loc+"optimal_solutions_discovered.csv","a")
            for b in opti:
                opt.write("Simulation,Solution,"+str(b[0][0].tolist())+",fitnesses,"+str(b[0].fitnesses.values)+",generation,"+str(b[1])+",discovered_at,"+str(end_time)+"\n")
            opt.close()
        ocount = len(optimals)
    return end_time, time_taken, population

def simulation_evaluate_population(pop,loc,fitness_type):
    global exit_pop_queue, pop_queue, pop_queue_lock, sim_evaluation, sim_fitnesses
    n = len(pop)
    sim_evaluation = [0.0]*n
    sim_fitnesses = [0.0]*n
    pop_queue = queue.Queue(n)
    exit_pop_queue = 0
    threads = []
    pop_queue_lock.acquire()
    count = 0
    for a in pop:
        pop_queue.put(a+[count])
        count += 1
    pop_queue_lock.release()
    for b in range(GPUS_AVAILABLE):
        thread = PopulationQueueThread(GPUS_AVAILABLE, b, pop_queue, loc, fitness_type)
        thread.start()
        threads.append(thread)
    while not pop_queue.empty():
        pass
    exit_pop_queue = 1
    for t in threads:
        t.join()
    sim_evaluation = list(map(list, zip(sim_evaluation,sim_fitnesses)))
    return sim_evaluation

class PopulationQueueThread(threading.Thread):
    def __init__(self, tn, device, q, loc, f):
        threading.Thread.__init__(self)
        self.tn = tn
        self.device = device
        self.q = q
        self.location = loc
        self.fitness_type = f
    def run(self):
        winexe = ""
        linexe = ""
        if self.fitness_type=="continuous":
            winexe = "simulation_executables\\PreyPredator_continuous_fitness.exe "
            linexe = "./simulation_executables/PreyPredator_console_continuous_fitness "
        else:
            winexe = "simulation_executables\\PreyPredator_discrete_fitness.exe "
            linexe = "./simulation_executables/PreyPredator_console_discrete_fitness "
        threadQueueFunction(self.tn, self.q, self.device, self.location, self.fitness_type, winexe, linexe)
        
def threadQueueFunction(tn, q, d, SP, ft, winexe, linexe):
    global gen, sim_evaluation, sim_fitnesses, pop_queue, pop_queue_lock, exit_pop_queue, gen
    #Make directories
    if not os.path.exists(SP+str(d)):
        os.makedirs(SP+str(d))
    if not os.path.exists(SP+str(d)+"/0.xml"):
        open(SP+str(d)+"/0.xml", "w").close()
    if not os.path.exists(SP+str(d)+"/save.csv"):
        open(SP+str(d)+"/save.csv", "w").close()
    while exit_pop_queue==0:
        pop_queue_lock.acquire()
        if not pop_queue.empty():
            x = q.get()
            ec = x[1]
            x = x[0]
            pop_queue_lock.release()
            open(SP+str(d)+"/spatially_complex_simulation_results.csv","w").close()
            if os.name=='nt':
                genComm = "simulation_executables\\xmlGenEx3.exe "+SP+str(d)+"/0.xml "+str(x[0])+" "+str(x[1])+" "+str(x[2])+" "+str(x[3])+" "+str(x[4])+" "+str(x[5])+" "+str(x[6])+" "+str(x[7])
                command = winexe+SP+str(d)+"/0.xml 1000 "+str(d)
            else:
                genComm = "./simulation_executables/xmlGen "+SP+str(d)+"/0.xml "+str(x[0])+" "+str(x[1])+" "+str(x[2])+" "+str(x[3])+" "+str(x[4])+" "+str(x[5])+" "+str(x[6])+" "+str(x[7])        
                command = linexe+SP+str(d)+"/0.xml 1000 "+str(d)
            os.system(genComm)
            os.system(command)
             #csv file data
            csv = open(SP+str(d)+"/spatially_complex_simulation_results.csv","r")
            li =  csv.readline()
            s = li.split(",")
            if len(s)<13:
                if ft=="continuous":
                    sim_evaluation[ec] = apply_continuous_fitness(-1000, -99, 99999999999)
                    sim_fitnesses[ec] = (-1000, -99, 99999999999)
                else:
                    sim_evaluation[ec] = apply_discrete_fitness(-1, -99, 99999999999)
                    sim_fitnesses[ec] = (-1, -99, 99999999999)
            else:
                if ft=="continuous":
                    sim_evaluation[ec] = apply_continuous_fitness(int(float(s[10])), int(float(s[11])),float(s[12]))
                else:
                    sim_evaluation[ec] = apply_discrete_fitness(int(float(s[10])), int(float(s[11])),float(s[12]))
                sim_fitnesses[ec] = (int(float(s[10])), int(float(s[11])),float(s[12]))
            csv.close()
            save = open(SP+str(d)+"/save.csv","a")
            save.write(li)
            save.close()
        else:
            pop_queue_lock.release()


def apply_continuous_fitness(prim,sec,ter):
    fitness = (0.001*prim)+(0.01*sec)-(0.00001*ter)
    return fitness,

def apply_discrete_fitness(prim,sec,ter):
    fitness = (1.0*prim)+(0.01*sec)-(0.00001*ter)
    return fitness,

#Create a new parameter set
def initialise_individual(container):
    global curr_pop
    new = [0]*9
    for i in range(len(parameter_limits)):
        if i<3 or i>4:
            new[i] = int(random.uniform(parameter_limits[i][0], parameter_limits[i][1]))
        else:
            new[i] = round(random.uniform(parameter_limits[i][0], parameter_limits[i][1]),6)
    new[8] = curr_pop
    curr_pop += 1
    new = np.array(new, dtype=np.float64).reshape(1,-1)
    return container(new)

def log(logbook, population, gen, nevals):
    global s1,s2,s3,s4
    record1 = s1.compile(population) if s1 else {}
    record2 = s2.compile(population) if s2 else {}
    record3 = s3.compile(population) if s3 else {}
    record4 = s4.compile(population) if s4 else {}
    logbook.record(gen=gen, nevals=nevals, **record1, **record2, **record3, **record4)
    return

def favour_offspring(parents, offspring, MU):
    choice = (list(zip(parents, [0]*len(parents))) +
              list(zip(offspring, [1]*len(offspring))))
    choice.sort(key=lambda x: ((x[0].fitness.values[0]), x[1]), reverse=True)
    return [x[0] for x in choice[:MU]], [x[0] for x in choice[:MU] if x[1]==1]

def select_parents(toolbox, individuals, k):
    parents = [random.choice(individuals) for i in range(k)]
    return [toolbox.clone(ind) for ind in parents]

#Mutate a parameter set, only reproduction rates and energy gain eligible for mutation
def mutate(toolbox, individual):
    #Heavily favour only a few mutated parameters, but leave small possibility for many mutations
    changes = np.random.choice([1,2,3,4,5,6,7,8], p=[0.35,0.25,0.15,0.1,0.05,0.05,0.025,0.025])
    ch = random.sample([0,1,2,3,4,5,6,7], changes)
    individual = individual[0]
    for c in ch:
        if c<3 or c>4:
            individual[c] += individual[c]+int(individual[c]*random.uniform(-MUTATION,MUTATION))
        else:
            individual[c] += individual[c]+(individual[c]*random.uniform(-MUTATION,MUTATION))
    for i in range(len(individual)-1):
        if individual[i]<parameter_limits[i][0]:
            individual[i] = parameter_limits[i][0]
        if individual[i]>parameter_limits[i][1]:
            individual[i] = parameter_limits[i][1]
    return individual,

#Mate two parameter sets. Offspring takes after one parent but with crossed over reproduction rates and energy gains
def mate(c,p1,p2):
    count = 0
    if len(p1)!=len(p2):
        print("Parents don't contain the same number of elements somehow",p1,p2)
    else:
        for e in range(len(p1)):
            if count>2 and count<7:
                if count<5:
                    p1[e] = ((p1[e]+p2[e])/2)
                else:
                    p1[e] = (int((p1[e]+p2[e])/2))
            else:
                parent = random.uniform(0,1)
                if parent>=c:
                    p1[e] = p2[e]
    return



########################### BATCH SIM #################################
def batch_simulation(num,time,simtype,seed,loc):
    global batch_queue_lock, batch_queue, exit_batch_queue, batch_times
    #Inititalise queue
    if not os.path.exists(loc+"/"+simtype+"/"):
        os.mkdir(loc+"/"+simtype+"/")
    nloc = loc+"/"+simtype+"/"
    batch_queue = queue.Queue(num)
    exit_batch_queue = 0
    threads = []
    batch_queue_lock.acquire()
    #Generate max number of parameters to evaluate
    for s in range(num):
        batch_queue.put(generate_parameter_vector(parameter_limits))
    batch_queue_lock.release()
    #Check the current time
    start_time = datetime.datetime.now()
    #Use all available devices to progress through generated parameters
    for b in range(GPUS_AVAILABLE):
        thread = Batch(b,batch_queue,nloc,start_time,simtype)
        thread.start()
        threads.append(thread)
    #Continue batch simulations until the queue is emptied or time limit is overrun
    while not batch_queue.empty():
        pass
    exit_batch_queue = 1
    for t in threads:
        t.join()
    #Copy all data to current experimentation progress folder for access by later functions
    open(nloc+"batch_simulation_data.csv","w").close()
    for a in range(GPUS_AVAILABLE):
        c = open(nloc+str(a)+"/spatially_complex_simulation_results.csv","r")
        l = c.readlines()
        c.close()
        bd = open(nloc+"batch_simulation_data.csv","a")
        bd.writelines(l)
        bd.close()
    return

#Generate a new input parameter vector for the predator prey and grass model. Population max 5000, Energy gains max 200, reproduction chance max 0.25
def generate_parameter_vector(lim):
    new = [0]*len(lim)
    for i in range(len(lim)):
        if type(lim[i][0])==int:
            new[i] = str(int(random.uniform(lim[i][0],lim[i][1])))
        else:
            new[i] = str(round(random.uniform(lim[i][0],lim[i][1]),6))
    return new

#Class for batch simulation of randomly generated inputs
class Batch(threading.Thread):
    def __init__(self, d, q, sp, t, st):
        threading.Thread.__init__(self)
        self.device = d
        self.queue = q
        self.save = sp
        self.start_time = t
        self.fitness_type = st
    def run(self):
        winexe = ""
        linexe = ""
        if self.fitness_type=="continuous":
            winexe = "simulation_executables\\PreyPredator_continuous_fitness.exe "
            linexe = "./simulation_executables/PreyPredator_console_continuous_fitness "
        else:
            winexe = "simulation_executables\\PreyPredator_discrete_fitness.exe "
            linexe = "./simulation_executables/PreyPredator_console_discrete_fitness "
        run_batch_queue(self.device, self.queue, self.save, self.start_time, winexe, linexe)
        
#Look through queue of randomly generated inputs and simulate a new input on the device assigned to the current thread      
def run_batch_queue(d,q,s,start,winexe,linexe):
    global batch_queue, batch_queue_lock, exit_batch_queue, batch_times
    #Ensure that all required files exist
    if not os.path.exists(s+str(d)):
        os.makedirs(s+str(d))
    if not os.path.exists(s+str(d)+"/0.xml"):
        open(s+str(d)+"/0.xml", "w").close()
    #Batch simulate while there still exist unevaluated inputs
    while exit_batch_queue==0:
        batch_queue_lock.acquire()
        if not batch_queue.empty():
            x = q.get()
            batch_queue_lock.release()
            #Generate initial state from initial paramter vector, then simulate
            if os.name=='nt':
                genComm = "simulation_executables\\xmlGenEx3.exe "+s+str(d)+"/0.xml "+str(x[0])+" "+str(x[1])+" "+str(x[2])+" "+str(x[3])+" "+str(x[4])+" "+str(x[5])+" "+str(x[6])+" "+str(x[7])
                command = winexe+s+str(d)+"/0.xml 1000 "+str(d)
            else:
                genComm = "./simulation_executables/xmlGen "+s+str(d)+"/0.xml "+str(x[0])+" "+str(x[1])+" "+str(x[2])+" "+str(x[3])+" "+str(x[4])+" "+str(x[5])+" "+str(x[6])+" "+str(x[7])        
                command = linexe+s+str(d)+"/0.xml 1000 "+str(d)
            #Assumes model is one which appends results to a file
            os.system(genComm)
            os.system(command)
        else:
            batch_queue_lock.release()
        #Measure and update the total simulation time taken by the current thread
        current_time = datetime.datetime.now()
        time_taken = current_time-start
        batch_times[d] = float(((time_taken.seconds*1000000)+time_taken.microseconds)/1000000)
    return






####################################### TRAIN SURROGATES ###################
def train_regression_surrogate_model(data_sizes,training,start_time,maxtime,bloc,loc,m, sm=-1):
    trained_models = 0
    timer = 0
    index = 0
    name = ""
    if m==1:
        name = "primary"
        index = 0
    elif m==2:
        name = "secondary"
        index = 1
    elif m==3:
        name = "tertiary"
        index = 2
    else:
        print("invalid model, exiting")
        return trained_models
    print("Training regression networks for",name,"fitness prediction with max time",maxtime,"minutes")
    while (not trained_models==sm and timer<maxtime):
        for i in data_sizes:
            training_datasets = []
            bsd = open(bloc+"/batch_simulation_data.csv","r")
            data = bsd.readlines()
            bsd.close()
            generate_training_dataset(data,training,i,training_datasets,loc)
            data = pickle.load(open(loc+"data/"+str(i)+"/all_examples.p","rb"))
            X,Y = get_xy(data)
            train_regressor(X,[[a[index],a[-1]] for a in Y],name,training,False,trained_models,i,loc)
            ct = datetime.datetime.now()
            ctt = ct-start_time
            timer = int(ctt.seconds/60)
            trained_models += 1
            if timer>=maxtime:
                print("Exceeded specified time limit,",maxtime,"minutes. Exiting early")
                break
    return trained_models, ct

def generate_training_dataset(data,test_sizes,max_data,td,loc,c=0):
    SP = loc+"/data/"+str(max_data)+"/"
    if not os.path.exists(loc+"/data/"):
        os.mkdir(loc+"/data/")
    if not os.path.exists(SP):
        os.mkdir(SP)
    random.shuffle(data)
    data = data[:max_data]
    lcount=0
    positive_examples = []
    negative_examples = []
    positive_response = 0
    if c=="continuous":
        positive_response = 1050
    else:
        positive_response = 1
    for li in data:
        sp = li.rstrip().split(",")
        if sp[0]!="Seed" and sp[0]!="time_taken":
            params = np.zeros(8)
            fitnesses = [0,0,0]
            for i in range(len(params)):
                params[i] = float(sp[i+1])
            fitnesses[0] = float(sp[10])
            fitnesses[1] = float(sp[11])
            fitnesses[2] = float(sp[12])
            X = [params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]]
            if fitnesses[0]==positive_response:
                positive_examples.append([X,fitnesses])
            else:
                negative_examples.append([X,fitnesses])
            lcount+=1
        else:
            lcount+=1
    pickle.dump(positive_examples,open(SP+"/positive_examples.p","wb"))
    pickle.dump(negative_examples,open(SP+"/negative_examples.p","wb"))
    td.append(data)
    pickle.dump(data,open(SP+"/all_examples.p","wb")) 
    return

def get_xy(data):
    x = []
    y = []
    count = 0
    for l in data:
        sp = l.rstrip().split(",")
        x.append([float(sp[1]),float(sp[2]),float(sp[3]),float(sp[4]),float(sp[5]),float(sp[6]),float(sp[7]),float(sp[8]),count])
        y.append([float(sp[10]),float(sp[11]),float(sp[12]),count])
        count+=1
    return x,y

def train_regressor(X,Y,name,train_sizes,prin,x,size,loc):
    SP = loc+str(size)+"/"
    if not os.path.exists(SP):
        os.mkdir(SP)
    SP = SP+name+"/"
    if not os.path.exists(SP):
        os.mkdir(SP)
    hidden_layers = (100,)*2
    scaler = StandardScaler()
    ds = []
    max_score = [0,-99]
    current_score = [0,-99]
    for ts in train_sizes:
        if prin:
            print(name,"training percentage",ts)
            print()
            print(name,"Random data sampling, training data size:",ts)
        testing_size = (1.0-ts)
        trainX,testX,trainY,testY = train_test_split(X,Y, test_size=testing_size)
        trainX = [m[:-1] for m in trainX]
        trainY = [n[:-1] for n in trainY]
        testX = [o[:-1] for o in testX]
        testY = [q[:-1] for q in testY]
        if prin:
            print("Training (and validation) data:",len(trainX),"Testing data:",len(testX))
        if len(trainX)>0 and len(testX)>0:
            if len(trainY[0])==1:
                trainY = np.array(trainY).ravel()
            scaler = scaler.fit(trainX)
            trainX = scaler.transform(trainX)
            testX = scaler.transform(testX)
            mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, verbose=False, tol=0.000000001, early_stopping=True)
            mlp.fit(trainX,trainY)
            test_score = mlp.score(testX,testY)
            current_score = max_score
            if test_score>=max_score[1]:
                if prin:
                    print("New best network for ",name," Fitness discovered by random sampling at:",ts,"scoring:",test_score)
                pickle.dump(mlp,open(SP+"/best_multiple_"+name+".p","wb"))
                pickle.dump(scaler,open(SP+"/scaler_multiple_"+name+".p","wb"))
                max_score = [ts,test_score]
            if test_score==1.0:
                if prin:
                    print("New best network for ",name," Fitness with accuracy==1.0 discovered by random sampling at:",ts,"scoring:",test_score)
                pickle.dump(mlp,open(SP+"/multiple_"+name+"_1_"+str(ts)+".p","wb"))
                pickle.dump(scaler,open(SP+"/scaler_multiple_"+name+"_1_"+str(ts)+".p","wb"))
                max_score = current_score
            ds.append(["random",ts,len(trainX),test_score,max_score])
        if prin: 
            print()
    if not os.path.exists(SP+"surrogate_training_data"+str(x)+".csv"):
        surrogate_training_file = open(SP+"surrogate_training_data"+str(x)+".csv","w")
    else:
        surrogate_training_file = open(SP+"surrogate_training_data"+str(x)+".csv","a")
    for a in ds:
        surrogate_training_file.write(str(a)+"\n")
    surrogate_training_file.close()
    return ds

def train_classifier_surrogate(data_sizes,training,start_time,maxtime,bloc,loc, sm=-1):
    trained_models = 0
    timer = 0
    print("Training classifier network for discrete primary fitness prediction with max time",maxtime,"minutes")
    while (not trained_models==sm and timer<maxtime):
        for i in data_sizes:
            training_datasets = []
            bsd = open(bloc+"/batch_simulation_data.csv","r")
            data = bsd.readlines()
            bsd.close()
            generate_training_dataset(data,training,i,training_datasets,loc,1)
            data = pickle.load(open(loc+"data/"+str(i)+"/all_examples.p","rb"))
            X,Y = get_xy(data)
            train_classifier(X,[[a[0],a[-1]] for a in Y],"primary","undersample",training,"sampling_before",False,trained_models,loc)
            ct = datetime.datetime.now()
            ctt = ct-start_time
            timer = int(ctt.seconds/60)
            trained_models += 1
            if timer>=maxtime:
                print("Exceeded specified time limit,",maxtime,"minutes. Exiting early")
                break
    return trained_models, ct

def sample_data(X,Y,samp):
    sampled_X = []
    sampled_Y = []
    if samp=="undersample":
        rus = RandomUnderSampler(random_state=0)
        modY = [a[0] for a in Y]
        if 1.0 in modY and 0.0 in modY:
            sx, sy = rus.fit_sample(X,modY)
            for i in sx:
                for j in Y:
                    if i[-1]==j[-1]:
                        sampled_X.append(i)
                        sampled_Y.append(j)
#        else:
#            print("Data set does not contain both negative and positive examples.")
#            print(modY)
    elif samp=="oversample":
        ros = RandomOverSampler(random_state=0)
        modY = [a[0] for a in Y]
        if 1.0 in modY and 0.0 in modY:
            sx, sy = ros.fit_sample(X,modY)
            for i in sx:
                for j in Y:
                    if i[-1]==j[-1]:
                        sampled_X.append(i)
                        sampled_Y.append(j)
#        else:
#            print("Data set does not contain both negative and positive examples.")
#            print(modY)
    else:
        sampled_X = X
        sampled_Y = Y
    return sampled_X,sampled_Y

def train_classifier(X,Y,name,samp,train_sizes,samptype,prin,x,loc):
    SP = loc+name+"/"
    if not os.path.exists(SP):
        os.mkdir(SP)
    SP = SP+samptype+"/"
    if not os.path.exists(SP):
        os.mkdir(SP)
    SP = SP+samp+"/"
    if not os.path.exists(SP):
        os.mkdir(SP)
    hidden_layers = (100,)*2
    scaler = StandardScaler()
    ds = []
    max_score = [0,-99]
    current_score = [0,-99]
    for ts in train_sizes:
        if prin:
            print()
            print(samp," data sampling, training size:",ts)
        sampled_X = None
        sampled_Y = None
        trainX = None
        trainY = None
        testX = None
        testY = None
        if samptype=="sampling_before":
            if prin:
                print(name,"sampling BEFORE splitting data")
            sampled_X,sampled_Y = sample_data(X,Y,samp)
            testing_size = (1.0-ts)
            trainX,testX,trainY,testY = train_test_split(sampled_X,sampled_Y, test_size=testing_size)
            trainX = [m[:-1] for m in trainX]
            trainY = [n[:-1] for n in trainY]
#        elif samptype=="sampling_after":
#            if prin:
#                print("sampling AFTER splitting data")
#            testing_size = (1.0-ts)
#            trainX,testX,trainY,testY = train_test_split(X,Y, test_size=testing_size)
#            sampled_X,sampled_Y = sample_data(trainX,trainY,samp)
#            trainX = [m[:-1] for m in sampled_X]
#            trainY = [n[:-1] for n in sampled_Y]
        testX = [o[:-1] for o in testX]
        testY = [q[:-1] for q in testY]        
        tpos = 0
        tneg = 0
        for i in trainY:
            if i[0]==1.0:
                tpos+=1
            else:
                tneg+=1
        if prin:
            print("training positive examples:",tpos,", negative examples:",tneg)
        
        epos = 0
        eneg = 0
        for j in testY:
            if j[0]==1.0:
                epos+=1
            else:
                eneg+=1
        if prin:
            print("testing positive examples:",epos,", negative examples:",eneg)
            print("Training (and validation) data:",len(trainX),"Testing data:",len(testX))
        if len(trainX)>1 and len(testX)>0:
            trainY = np.array(trainY).ravel()
            scaler = scaler.fit(trainX)
            trainX = scaler.transform(trainX)
            testX = scaler.transform(testX)
            mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, verbose=False, tol=0.000000001, early_stopping=True)
            mlp.fit(trainX,trainY)
            test_score = mlp.score(testX,testY)
            current_score = max_score
            if test_score>=max_score[1]:
                if prin:
                    print("New best network for ",name," Fitness discovered by",samp,"at:",ts,"scoring:",test_score)
                pickle.dump(mlp,open(SP+"/best_multiple_"+name+".p","wb"))
                pickle.dump(scaler,open(SP+"/scaler_multiple_"+name+".p","wb"))
                max_score = [ts,test_score]
            if test_score==1.0:
                if prin:
                    print("New best network for ",name," Fitness with accuracy==1.0 discovered by",samp,"at:",ts,"scoring:",test_score)
                pickle.dump(mlp,open(SP+"/multiple_"+name+"_1_"+str(ts)+".p","wb"))
                pickle.dump(scaler,open(SP+"/scaler_multiple_"+name+"_1_"+str(ts)+".p","wb"))
                max_score = current_score
            ds.append([samp,ts,len(trainX),test_score,max_score])
        if prin:
            print()
    if not os.path.exists(SP+"surrogate_training_data"+str(x)+".csv"):
        surrogate_training_file = open(SP+"surrogate_training_data"+str(x)+".csv","w")
    else:
        surrogate_training_file = open(SP+"surrogate_training_data"+str(x)+".csv","a")
    for a in ds:
        surrogate_training_file.write(str(a)+"\n")
    surrogate_training_file.close()
    return ds



######################################## SURROGATE GA ##############################################
def surrogate_ga(x,sizes,m,l,g,time,mloc,loc,c,s=1):
    primary_loc = ""
    scaler_loc = ""
    if c=="continuous":
        primary_loc = "/primary/best_multiple_primary.p"
        scaler_loc = "/primary/scaler_multiple_primary.p"
    else:
        primary_loc = "/../primary/sampling_before/undersample/best_multiple_primary.p"
        scaler_loc = "/../primary/sampling_before/undersample/scaler_multiple_primary.p"
    timer = 0
    start_time = datetime.datetime.now()
    complete = False
    while (timer<time and not complete):
        for j in sizes:
            for i in range(x):
                nn0 = pickle.load(open(mloc+str(j)+primary_loc,"rb"))
                nn1 = pickle.load(open(mloc+str(j)+"/secondary/best_multiple_secondary.p","rb"))
                nn2 = pickle.load(open(mloc+str(j)+"/tertiary/best_multiple_tertiary.p","rb"))
                scaler0 = pickle.load(open(mloc+str(j)+scaler_loc,"rb"))
                scaler1 = pickle.load(open(mloc+str(j)+"/secondary/scaler_multiple_secondary.p","rb"))
                scaler2 = pickle.load(open(mloc+str(j)+"/tertiary/scaler_multiple_tertiary.p","rb"))
                run_surrogate_ga(m,l,g,time,str(j)+"/",[nn0,nn1,nn2],[scaler0,scaler1,scaler2],c,loc)
                ct = datetime.datetime.now()
                ctt = ct-start_time
                timer = int(ctt.seconds/60)
                if timer>=time:
                    print("Exceeded specified time limit,",time,"minutes. Exiting early")
                    break
        complete = True
    return ct

def run_surrogate_ga(m,l,g,time,name,networks,scalers,c,loc,s=1):
    global curr_pop, gen, s1, s2, s3, s4, population, toolbox, logbook
    if not os.path.exists(loc+name):
        os.mkdir(loc+name)
    if not os.path.exists(loc+name+"runs/"):
        os.mkdir(loc+name+"runs/")
    loc = loc+name
    runloc = loc+"runs/"
    curr_pop = 0
    #Initialise GA tools
    creator.create("Fitness", base.Fitness, weights=(1.0,))
    if c=="continuous":
        creator.create("Fitnesses", base.Fitness, weights=(0.001,0.01,-0.00001))
    else:
        creator.create("Fitnesses", base.Fitness, weights=(1.0,0.01,-0.00001))
    creator.create("Individual", list, fitness=creator.Fitness, fitnesses=creator.Fitnesses)
    toolbox = base.Toolbox()
    toolbox.register("select_parents", select_parents, toolbox)
    toolbox.register("mutate", mutate, toolbox)
    toolbox.register("mate", mate, 0.5)
    toolbox.register("individual", initialise_individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #Initialise Logging tools
    s1 = tools.Statistics(lambda ind: ind.fitness.values[0])
    s1.register("avg", np.mean)
    s1.register("std", np.std)
    s1.register("min", np.min)
    s1.register("max", np.max)
    s2 = tools.Statistics(lambda ind: ind.fitnesses.values[0])
    s2.register("p-avg", np.mean)
    s2.register("p-std", np.std)
    s2.register("p-min", np.min)
    s2.register("p-max", np.max)
    s3 = tools.Statistics(lambda ind: ind.fitnesses.values[1])
    s3.register("s-avg", np.mean)
    s3.register("s-std", np.std)
    s3.register("s-min", np.min)
    s3.register("s-max", np.max)
    s4 = tools.Statistics(lambda ind: ind.fitnesses.values[2])
    s4.register("t-avg", np.mean)
    s4.register("t-std", np.std)
    s4.register("t-min", np.min)
    s4.register("t-max", np.max)
    end_pop = None
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (s1.fields+s2.fields+s3.fields+s4.fields if s1 and s2 and s3 and s4 else [])
    population = toolbox.population(n=m)
    eval_count = len(population)
    initial_fitnesses = surrogate_evaluate_population(population,networks,scalers,c)
    unique_run_seed = random.randrange(sys.maxsize)
    pr = runloc+str(unique_run_seed)+".csv"
    if not os.path.exists(pr):
        population_record = open(pr,"w")
    else:
        population_record = open(runloc+str(unique_run_seed)+"(1).csv","w")
    population_record.write("Surrogate,generation,0,mu,"+str(m)+",lambda,"+str(l)+"\n")
    for i in range(len(initial_fitnesses)):
        population[i].fitness.values = initial_fitnesses[i][0]
        population[i].fitnesses.values = initial_fitnesses[i][1]
        population_record.write("\t")
        for j in population[i][0].tolist():
            population_record.write(str(j)+",")
        population_record.write("fitness,("+str(population[i].fitness.values[0].tolist()[0])+"),fitnesses,("+str(population[i].fitnesses.values[0].tolist()[0])+","+str(population[i].fitnesses.values[1].tolist()[0])+","+str(population[i].fitnesses.values[2].tolist()[0])+","+")\n")    
    population_record.close()
    gen = 0
    log(logbook, population, gen, len(population))
    end_time, end_pop = surrogate_pred_prey_ga(m,l,g,eval_count,population,loc,runloc,pr,unique_run_seed,networks,scalers,c)
    if c=="continuous":
        if os.path.exists(loc+"continuous_surrogate_ga_performances.csv"):
            pf = open(loc+"continuous_surrogate_ga_performances.csv","a")
        else:
            pf = open(loc+"continuous_surrogate_ga_performances.csv","w")
        pf.write(str(logbook)+"\n")
        pf.close()
    else:
        if os.path.exists(loc+"discrete_surrogate_ga_performances.csv"):
            pf = open(loc+"discrete_surrogate_ga_performances.csv","a")
        else:
            pf = open(loc+"discrete_surrogate_ga_performances.csv","w")
        pf.write(str(logbook)+"\n")
        pf.close()
    return

def surrogate_evaluate_population(pop,networks,scalers,c):
    global evaluation, fitnesses, nn,nn0,nn1,nn2, scaler0,scaler1,scaler2
    n = len(pop)
    evaluation = [0.0]*n
    fitnesses = [0.0]*n
    count = 0
    for x in pop:
        yvals = []
        if len(networks)>1:
            for i in range(len(networks)):
                xcurr = scalers[i].transform(x[0][:-1].reshape(1,-1))
                yvals.append(networks[i].predict(xcurr))
            
        else:
            xcurr = scalers[0].transform(x[0][:-1].reshape(1,-1))
            yvals = networks[0].predict(xcurr)[0]
        if c=="continuous":
            evaluation[count] = apply_continuous_fitness(yvals[0],yvals[1],yvals[2])
        else:
            evaluation[count] = apply_discrete_fitness(yvals[0],yvals[1],yvals[2])
        fitnesses[count] = (yvals[0],yvals[1],yvals[2])
        count+=1
    evaluation = list(map(list, zip(evaluation,fitnesses)))
    return evaluation

def surrogate_pred_prey_ga(m,l,g,eval_count,population,loc,runloc,pr,seed,networks,scalers,ft):
    global curr_pop,toolbox,logbook
    gen = 0
    conditions_met = 0
    optimals = []
    ocount = 0
    while(gen<g and conditions_met==0):
        gen += 1
        nevals = 0
        curr_pop = 0
        #Generate offspring
        offspring = []
        evaluations = []
        #crossover is being used, it is done before mutation
        for i in range(l):
            new = random.uniform(0,1)
            if new<0.5:
                p1 = toolbox.individual()
            else:
                p1, p2 = [toolbox.clone(x) for x in toolbox.select_parents(population, 2)]
                toolbox.mate(p1, p2)
            offspring.append(p1)
        #Mutate
        for off in offspring:
            off, = toolbox.mutate(off)
        nevals += len(offspring)
        evaluations = surrogate_evaluate_population(offspring,networks,scalers,ft)
        for i in range(len(evaluations)):
            offspring[i].fitness.values = evaluations[i][0]
            offspring[i].fitnesses.values = evaluations[i][1]
        eval_count += nevals
        # Select the next generation, favouring the offspring in the event of equal fitness values
        population = surrogate_favour_offspring(population, offspring, m)
        #Print a report about the current generation
        if nevals > 0:
            log(logbook, population, gen, nevals)
            #Save to file in case of early exit
        log_fitness = open(loc+"fitness_log.csv","w")
        log_fitness.write(str(logbook)+"\n")
        log_fitness.close()
        if os.path.exists(pr):
            population_record = open(pr,"a")
        else:
            population_record = open(runloc+seed+"(1).csv","a")
        check_nonunique = []
        for p in population:
            population_record.write("\t")
            for q in p[0].tolist():
                population_record.write(str(q)+",")
            population_record.write("fitness,("+str(p.fitness.values[0].tolist()[0])+"),fitnesses,("+str(p.fitnesses.values[0].tolist()[0])+","+str(p.fitnesses.values[1].tolist()[0])+","+str(p.fitnesses.values[2].tolist()[0])+","+")\n")
            if ft==1 and p.fitness.values[0]>0.95 and p.fitnesses.values[0]>900 and p.fitnesses.values[2]<15000:
                for opt in optimals:
                    check_nonunique.append(all(elem in p[0][:-1] for elem in opt[0][:-1]))
                if not any(check_nonunique):
                    optimals.append((p,gen))
            if ft==0 and p.fitness.values[0]>0.95 and p.fitnesses.values[0]>0 and p.fitnesses.values[2]<15000:
                for opt in optimals:
                    check_nonunique.append(all(elem in p[0][:-1] for elem in opt[0][:-1]))
                if not any(check_nonunique):
                    optimals.append((p,gen))
        population_record.write("Surrogate,generation,"+str(gen)+"\n")
        population_record.close()
        end_time = datetime.datetime.now()
        opti = optimals[ocount:]
        if len(opti)>0:
            opt = open(loc+"optimal_solutions_discovered.csv","a")
            for b in opti:
                wri = [str(b[0].fitnesses.values[x].tolist()[0])+"," for x in range(len(b[0].fitnesses.values))]
                writ = ""
                writ = writ.join(wri)[:-1]
                opt.write("Surrogate,Solution,"+str(b[0][0].tolist())+",fitnesses,"+writ+",generation,"+str(b[1])+",discovered_at,"+str(end_time)+"\n")
            opt.close()
        ocount = len(optimals)
    return end_time, population

def surrogate_favour_offspring(parents, offspring, MU):
    choice = (list(zip(parents, [0]*len(parents))) +
              list(zip(offspring, [1]*len(offspring))))
    choice.sort(key=lambda x: ((x[0].fitness.values[0]), x[1]), reverse=True)
    return [x[0] for x in choice[:MU]]
#
########################## SURRSIM COMPARISON ############################################
def surrogate_ga_sim_eval(x,sizes,m,l,g,time,loc,mloc,c):
    primary_loc = ""
    scaler_loc = ""
    if c=="continuous":
        primary_loc = "/primary/best_multiple_primary.p"
        scaler_loc = "/primary/scaler_multiple_primary.p"
    else:
        primary_loc = "/../primary/sampling_before/undersample/best_multiple_primary.p"
        scaler_loc = "/../primary/sampling_before/undersample/scaler_multiple_primary.p"
    timer = 0
    start_time = datetime.datetime.now()
    complete = False
    while (timer<time and not complete):
        for j in sizes:
            for i in range(x):
                nn0 = pickle.load(open(mloc+str(j)+primary_loc,"rb"))
                nn1 = pickle.load(open(mloc+str(j)+"/secondary/best_multiple_secondary.p","rb"))
                nn2 = pickle.load(open(mloc+str(j)+"/tertiary/best_multiple_tertiary.p","rb"))
                scaler0 = pickle.load(open(mloc+str(j)+scaler_loc,"rb"))
                scaler1 = pickle.load(open(mloc+str(j)+"/secondary/scaler_multiple_secondary.p","rb"))
                scaler2 = pickle.load(open(mloc+str(j)+"/tertiary/scaler_multiple_tertiary.p","rb"))
                surr_sim_ga(m,l,g,time,str(j)+"/",[nn0,nn1,nn2],[scaler0,scaler1,scaler2],c,loc)
                ct = datetime.datetime.now()
                ctt = ct-start_time
                timer = int(ctt.seconds/60)
                if timer>=time:
                    print("Exceeded specified time limit,",time,"minutes. Exiting early")
                    break
        complete = True
    return ct
    
def surr_sim_ga(m,l,g,time,name,networks,scalers,c,loc,s=1):
    global curr_pop, gen, s1, s2, s3, s4, population, pop2, toolbox, logbook, logbook2
    if not os.path.exists(loc+name):
        os.mkdir(loc+name)
    if not os.path.exists(loc+name+"runs/"):
        os.mkdir(loc+name+"runs/")
    loc = loc+name
    runloc = loc+"runs/"
    curr_pop = 0
    creator.create("Fitness", base.Fitness, weights=(1.0,))
    if c=="continuous":
        creator.create("Fitnesses", base.Fitness, weights=(0.001,0.01,-0.00001))
    else:
        creator.create("Fitnesses", base.Fitness, weights=(1.0,0.01,-0.00001))
    creator.create("Individual", list, fitness=creator.Fitness, fitnesses=creator.Fitnesses)
    toolbox = base.Toolbox()
    toolbox.register("select_parents", select_parents, toolbox)
    toolbox.register("mutate", mutate, toolbox)
    toolbox.register("mate", mate, 0.5)
    toolbox.register("individual", initialise_individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    s1 = tools.Statistics(lambda ind: ind.fitness.values[0])
    s1.register("avg", np.mean)
    s1.register("std", np.std)
    s1.register("min", np.min)
    s1.register("max", np.max)
    s2 = tools.Statistics(lambda ind: ind.fitnesses.values[0])
    s2.register("p-avg", np.mean)
    s2.register("p-std", np.std)
    s2.register("p-min", np.min)
    s2.register("p-max", np.max)
    s3 = tools.Statistics(lambda ind: ind.fitnesses.values[1])
    s3.register("s-avg", np.mean)
    s3.register("s-std", np.std)
    s3.register("s-min", np.min)
    s3.register("s-max", np.max)
    s4 = tools.Statistics(lambda ind: ind.fitnesses.values[2])
    s4.register("t-avg", np.mean)
    s4.register("t-std", np.std)
    s4.register("t-min", np.min)
    s4.register("t-max", np.max)
    end_pop = None
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (s1.fields+s2.fields+s3.fields+s4.fields if s1 and s2 and s3 and s4 else [])
    logbook2 = tools.Logbook()
    logbook2.header = ['gen', 'nevals'] + (s1.fields+s2.fields+s3.fields+s4.fields if s1 and s2 and s3 and s4 else [])
    gen = 0
    population = toolbox.population(n=m)
    pop2 = copy.deepcopy(population)
    eval_count = len(population)
    initial_fitnesses = surrogate_evaluate_population(population,networks,scalers,c)
    unique_run_seed = random.randrange(sys.maxsize)
    pr = runloc+str(unique_run_seed)+".csv"
    if not os.path.exists(pr):
        population_record = open(pr,"w")
    else:
        population_record = open(runloc+str(unique_run_seed)+"(1).csv","w")
    population_record.write("Surrogate,generation,0,mu,"+str(m)+",lambda,"+str(l)+"\n")
    for i in range(len(initial_fitnesses)):
        population[i].fitness.values = initial_fitnesses[i][0]
        population[i].fitnesses.values = initial_fitnesses[i][1]
        population_record.write("\t")
        for j in population[i][0].tolist():
            population_record.write(str(j)+",")
        population_record.write("fitness,("+str(population[i].fitness.values[0].tolist()[0])+"),fitnesses,("+str(population[i].fitnesses.values[0].tolist()[0])+","+str(population[i].fitnesses.values[1].tolist()[0])+","+str(population[i].fitnesses.values[2].tolist()[0])+","+")\n")
    sim_eval_fitnesses_initial = simulation_evaluate_population(pop2,loc,c)
    population_record.write("Simulation,generation,0,mu,"+str(m)+",lambda,"+str(l)+"\n")
    for i in range(len(sim_eval_fitnesses_initial)):
        pop2[i].fitness.values = sim_eval_fitnesses_initial[i][0]
        pop2[i].fitnesses.values = sim_eval_fitnesses_initial[i][1]
        population_record.write("\t")
        for j in pop2[i][0].tolist():
            population_record.write(str(j)+",")
        population_record.write("fitness,"+str(pop2[i].fitness.values)+",fitnesses,"+str(pop2[i].fitnesses.values)+"\n")   
    population_record.close()    
    log(logbook, population, gen, len(population))
    log(logbook2, pop2, gen, len(pop2))
    end_time, end_pop = surrsim_pred_prey_ga(m,l,g,eval_count,population,loc,runloc,pr,unique_run_seed,networks,scalers,c)
    if c=="continuous":
        if os.path.exists(loc+"continuous_surrogate_ga_performances.csv"):
            pf = open(loc+"continuous_surrogate_ga_performances.csv","a")
        else:
            pf = open(loc+"continuous_surrogate_ga_performances.csv","w")
        pf.write(str(logbook)+"\n")
        pf.close()
        if os.path.exists(loc+"continuous_surrogate_ga_simulation_evaluation.csv"):
            pf = open(loc+"continuous_surrogate_ga_simulation_evaluation.csv","a")
        else:
            pf = open(loc+"continuous_surrogate_ga_simulation_evaluation.csv","w")
        pf.write(str(logbook2)+"\n")
        pf.close()
    else:
        if os.path.exists(loc+"discrete_surrogate_ga_performances.csv"):
            pf = open(loc+"discrete_surrogate_ga_performances.csv","a")
        else:
            pf = open(loc+"discrete_surrogate_ga_performances.csv","w")
        pf.write(str(logbook)+"\n")
        pf.close()
        if os.path.exists(loc+"discrete_surrogate_ga_simulation_evaluation.csv"):
            pf = open(loc+"discrete_surrogate_ga_simulation_evaluation.csv","a")
        else:
            pf = open(loc+"discrete_surrogate_ga_simulation_evaluation.csv","w")
        pf.write(str(logbook2)+"\n")
        pf.close()
    return
    
def surrsim_pred_prey_ga(m,l,g,eval_count,population,loc,runloc,pr,seed,networks,scalers,c):
    global curr_pop, toolbox, logbook, gen, pop2
    gen = 0
    conditions_met = 0
    optimals  = []
    opt2 = []
    ocount = 0
    ocount2 = 0
    while(gen<g and conditions_met==0):
        gen += 1
        nevals = 0
        curr_pop = 0
        #Generate offspring
        offspring = []
        evaluations = []
        #crossover is being used, it is done before mutation
        for i in range(l):
            new = random.uniform(0,1)
            if new<0.5:
                p1 = toolbox.individual()
            else:
                p1, p2 = [toolbox.clone(x) for x in toolbox.select_parents(population, 2)]
                toolbox.mate(p1, p2)
            offspring += [p1]
        #Mutate
        for off in offspring:
            off, = toolbox.mutate(off)
        nevals += len(offspring)
        evaluations = surrogate_evaluate_population(offspring,networks,scalers,c)
        for i in range(len(evaluations)):
            offspring[i].fitness.values = evaluations[i][0]
            offspring[i].fitnesses.values = evaluations[i][1]
        eval_count += nevals
        # Select the next generation, favouring the offspring in the event of equal fitness values
        population, new_individuals = surrsim_favour_offspring(population, offspring, m)        
        ni = copy.deepcopy(new_individuals)
        new_indiv_sim_fit = simulation_evaluate_population(ni,loc,c)
        for i in range(len(new_indiv_sim_fit)):
            ni[i].fitness.values = new_indiv_sim_fit[i][0]
            ni[i].fitnesses.values = new_indiv_sim_fit[i][1]
        pop_curr = copy.deepcopy(population)
        for i in range(len(pop_curr)):
            setted = set(pop_curr[i][0].tolist())
            for j in range(len(ni)):
                if setted==set(ni[j][0].tolist()):
                    pop_curr[i].fitness.values = ni[j].fitness.values
                    pop_curr[i].fitnesses.values = ni[j].fitnesses.values
            for k in pop2:
                if setted==set(k[0].tolist()):
                    pop_curr[i].fitness.values = k.fitness.values
                    pop_curr[i].fitnesses.values = k.fitnesses.values
        pop2 = copy.deepcopy(pop_curr)
        if nevals > 0:
            log(logbook, population, gen, nevals)
            log(logbook2, pop2, gen, nevals)
            #Save to file in case of early exit
        log_fitness = open(loc+"fitness_log.csv","w")
        log_fitness.write(str(logbook)+"\n")
        log_fitness.write(str(logbook2))
        log_fitness.close()
        if os.path.exists(pr):
            population_record = open(pr,"a")
        else:
            population_record = open(runloc+seed+"(1).csv","a")
        check_nonunique = []
        check_simulation_nonunique = []
        population_record.write("Surrogate,generation,"+str(gen)+"\n")
        for p in population:
            population_record.write("\t")
            for q in p[0].tolist():
                population_record.write(str(q)+",")
            population_record.write("fitness,("+str(p.fitness.values[0].tolist()[0])+"),fitnesses,("+str(p.fitnesses.values[0].tolist()[0])+","+str(p.fitnesses.values[1].tolist()[0])+","+str(p.fitnesses.values[2].tolist()[0])+","+")\n")
            if c=="continuous" and p.fitness.values[0]>0.95 and p.fitnesses.values[0]>900 and p.fitnesses.values[2]<15000:
                for opt in optimals:
                    check_nonunique.append(all(elem in p[0][:-1] for elem in opt[0][:-1]))
                if not any(check_nonunique):
                    optimals.append((p,gen))
            if c=="discrete" and p.fitness.values[0]>0.95 and p.fitnesses.values[0]>0 and p.fitnesses.values[2]<15000:
                for opt in optimals:
                    check_nonunique.append(all(elem in p[0][:-1] for elem in opt[0][:-1]))
                if not any(check_nonunique):
                    optimals.append((p,gen))
        population_record.write("Simulation,generation,"+str(gen)+"\n")
        for r in pop2:
            population_record.write("\t")
            for s in r[0].tolist():
                population_record.write(str(s)+",")
            population_record.write("fitness,"+str(r.fitness.values)+",fitnesses,"+str(r.fitnesses.values)+"\n")
            if c=="continuous" and r.fitness.values[0]>0.95 and r.fitnesses.values[0]>900 and r.fitnesses.values[2]<15000:
                for opt in opt2:
                    check_simulation_nonunique.append(all(elem in r[0][:-1] for elem in opt[0][:-1]))
                if not any(check_simulation_nonunique):
                    opt2.append((r,gen))
            if c=="discrete" and r.fitness.values[0]>0.95 and r.fitnesses.values[0]>0 and r.fitnesses.values[2]<15000:
                for opt in opt2:
                    check_simulation_nonunique.append(all(elem in r[0][:-1] for elem in opt[0][:-1]))
                if not any(check_simulation_nonunique):
                    opt2.append((r,gen))
        population_record.close()
        end_time = datetime.datetime.now()        
        opti = optimals[ocount:]
        if len(opti)>0:
            opt = open(loc+"optimal_solutions_discovered.csv","a")
            for b in opti:
                wri = [str(b[0].fitnesses.values[x].tolist()[0])+"," for x in range(len(b[0].fitnesses.values))]
                writ = ""
                writ = writ.join(wri)[:-1]
                opt.write("Surrogate,Solution,"+str(b[0][0].tolist())+",fitnesses,"+writ+",generation,"+str(b[1])+",discovered_at,"+str(end_time)+"\n")
            opt.close()
        ocount = len(optimals)
        opti2 = opt2[ocount2:]
        if len(opti2)>0:
            opt = open(loc+"optimal_solutions_discovered.csv","a")
            for b in opti2:
                    opt.write("Simulation,Solution,"+str(b[0][0].tolist())+",fitnesses,"+str(b[0].fitnesses.values)+",generation,"+str(b[1])+",discovered_at,"+str(end_time)+"\n")
            opt.close()
        ocount2 = len(opt2)
    return end_time, population
    
def surrsim_favour_offspring(parents, offspring, MU):
    choice = (list(zip(parents, [0]*len(parents))) +
              list(zip(offspring, [1]*len(offspring))))
    choice.sort(key=lambda x: ((x[0].fitness.values[0]), x[1]), reverse=True)
    return [x[0] for x in choice[:MU]], [x[0] for x in choice[:MU] if x[1]==1]
