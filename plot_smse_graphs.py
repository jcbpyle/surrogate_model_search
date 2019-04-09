# -*- coding: utf-8 -*-
"""
Created on Thu Mar 7 12:50:43 2019

@author: James
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context(rc={"lines.linewidth": 4})
sns.set_style("whitegrid")

def simulation_ga_graphs(cont_loc,disc_loc):
    print("Graphing GA performance and variance.")
    data2,l2,mu2,lam2 = get_simga_data(0,disc_loc)
    data,l,mu,lam = get_simga_data(0,cont_loc)
    check = [data[1],data2[1],data[2],data2[2],data[3],data2[3],data[4],data2[4],data[5],data2[5],data[6],data2[6],data[7],data2[7],data[8],data2[8]]
    if any([len(x)<len(data[0]) for x in check]):
        data[0] = data[0][:min([len(x) for x in check])]
    if any([len(x)<len(data2[0]) for x in check]):
        data2[0] = data2[0][:min([len(x) for x in check])]
    #Plot average and average standard deviation error bars at every generation
    plt_std_simga([data[0],data2[0]],[data[1],data2[1]],[data[2],data2[2]],"Simulations Performed","Evaluated Fitness",["Continuous Primary Fitness\n("+str(l)+" runs)","Discrete Primary Fitness\n("+str(l2)+" runs)"],"Increasing Fitness of Simulation Guided ("+str(MU)+"+"+str(LAMBDA)+")GA",IN_PROGRESS+"/graphs/sim_ga_average")
    print("Graphs completed. Saved to "+IN_PROGRESS+".\n")
    return

def get_simga_data(sd,f):
    fitness_av_sim = []
    fitness_std_sim = []
    primary_av_sim = []
    primary_std_sim = []
    secondary_av_sim = []
    secondary_std_sim = []
    tertiary_av_sim = []
    tertiary_std_sim = []
    sims = []
    if not os.path.exists(IN_PROGRESS+f):
        print("please run multiple seeded GA before calling this function. seeded_ga_performances.csv must exist")
        exit(1)
    rf = open(IN_PROGRESS+f,"r")
    lcount2 = -1
    mcount = 0
    m = 0
    l = 0
    scount = sd
    #Check all GA's performance
    for li in rf:
        s = li.rstrip().replace(" ","").split("\t")
        #If not new GA
        if not s[0]=="gen":
            if mcount==0:
                m = int(s[1])
                mcount+=1
            elif mcount==1:
                l = int(s[1])
                mcount+=1
            
            if not len(sims)==GENERATIONS+1:
                    scount += int(s[1])
                    sims.append(scount)
            
            if not (float(s[8])==-1000 or float(s[8])==-1):
                fitness_av_sim[lcount2].append(float(s[2]))
                fitness_std_sim[lcount2].append(float(s[3]))
                primary_av_sim[lcount2].append(float(s[6]))
                primary_std_sim[lcount2].append(float(s[7]))
                secondary_av_sim[lcount2].append(float(s[10]))
                secondary_std_sim[lcount2].append(float(s[11]))
                tertiary_av_sim[lcount2].append(float(s[14]))
                tertiary_std_sim[lcount2].append(float(s[15]))
            else:
                fitness_av_sim[lcount2].append(-10)
                fitness_std_sim[lcount2].append(1)
                primary_av_sim[lcount2].append(-1000)
                primary_std_sim[lcount2].append(1)
                secondary_av_sim[lcount2].append(-100)
                secondary_std_sim[lcount2].append(1)
                tertiary_av_sim[lcount2].append(1000000)
                tertiary_std_sim[lcount2].append(1)
        else:            
            fitness_av_sim.append([])
            fitness_std_sim.append([])
            primary_av_sim.append([])
            primary_std_sim.append([])
            secondary_av_sim.append([])
            secondary_std_sim.append([])
            tertiary_av_sim.append([])
            tertiary_std_sim.append([])
            lcount2 += 1
            if not len(sims)==GENERATIONS+1:
                sims = []
            scount = sd
     
    for p in range(len(primary_av_sim)):
        if any([x>=10 for x in primary_av_sim[p]]):
            primary_av_sim[p] = [float(x)/1000 for x in primary_av_sim[p]]
            primary_std_sim[p] = [float(x)/1000 for x in primary_std_sim[p]]
           
    fitness_av_sim = [x for x in fitness_av_sim if len(x)==GENERATIONS+1]
    for index in range(len(fitness_av_sim[0])):
        testerino = [x[index] for x in fitness_av_sim]
        if -10 in testerino:
            for i in fitness_av_sim:
                i[index] = -0.1
                
    primary_av_sim = [x for x in primary_av_sim if len(x)==GENERATIONS+1]
    for index in range(len(primary_av_sim[0])):
        testerino = [x[index] for x in primary_av_sim]
        if -1000 in testerino:
            for i in primary_av_sim:
                i[index] = -1
                                
    secondary_av_sim = [x for x in secondary_av_sim if len(x)==GENERATIONS+1]
    for index in range(len(secondary_av_sim[0])):
        testerino = [x[index] for x in secondary_av_sim]
        if -100 in testerino:
            for i in secondary_av_sim:
                i[index] = -1
            
    tertiary_av_sim = [x for x in tertiary_av_sim if len(x)==GENERATIONS+1]
    for index in range(len(tertiary_av_sim[0])):
        testerino = [x[index] for x in tertiary_av_sim]
        if 100000 in testerino:
            for i in tertiary_av_sim:
                i[index] = -100
                
    data = len(primary_av_sim)
    
    #Find average and average standard deviation at each of the GA generations for plotting
    if len(fitness_std_sim)>0:
        fstd2 = fitness_std_sim[0]
        fav2 = [sum(x)/len(fitness_av_sim) for x in zip(*fitness_av_sim)]
        zippedf2 = [x for x in zip(*fitness_std_sim)]
        if len(zippedf2[0])>1:
            fstd2 = np.std(np.array(zippedf2),axis=1)
    if len(primary_std_sim)>0:
        pstd2 = primary_std_sim[0]
        pav2 = [sum(x)/len(primary_av_sim) for x in zip(*primary_av_sim)]
        zippedp2 = [x for x in zip(*primary_std_sim)]
        if len(zippedp2[0])>1:
            pstd2 = np.std(np.array(zippedp2),axis=1)
    if len(secondary_std_sim)>0:
        sstd2 = secondary_std_sim[0]
        sav2 = [sum(x)/len(secondary_av_sim) for x in zip(*secondary_av_sim)]
        zippeds2 = [x for x in zip(*secondary_std_sim)]
        if len(zippeds2[0])>1:
            sstd2 = np.std(np.array(zippeds2),axis=1)
    if len(tertiary_std_sim)>0:
        tstd2 = tertiary_std_sim[0]
        tav2 = [sum(x)/len(tertiary_av_sim) for x in zip(*tertiary_av_sim)]
        zippedt2 = [x for x in zip(*tertiary_std_sim)]
        if len(zippedt2[0])>1:
            tstd2 = np.std(np.array(zippedt2),axis=1)
    return [sims,fav2,fstd2,pav2,pstd2,sav2,sstd2,tav2,tstd2],data,m,l

def plt_std_simga(xlist,ylist,stdlist,xl,yl,l,title,save):
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=24, color='black')
    plt.xlabel(xl, fontsize=18)
    plt.ylabel(yl, fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    linest = ['-','--']
    elinst = ['--',':']
    for i in range(len(xlist)):
        eb = plt.errorbar(xlist[i],ylist[i], yerr=stdlist[i],  label=l[i], capsize=2*(i+1), ls=linest[i])
        eb[-1][0].set_linestyle(elinst[i])
        eb[-1][0].set_linewidth(abs(2-i)*0.5)
    plt.legend(frameon=True, fontsize=18, loc=4)
    plt.savefig(save+"_std.png", bbox_inches='tight')
    plt.savefig(save+"_std.pdf", bbox_inches='tight')
    plt.show()
    plt.close()
    return

def average_sim_ga_time(loc):
    times = open(loc,"r")
    data = times.readlines()
    times.close()
    time_list = []
    for i in data:
        sp = i.rstrip().split(",")
        ti = sp[8].split(":")
        ms = ti[2].split(".")
        time_list.append((int(ti[0])*3600)+(int(ti[1])*60)+(int(ms[0]))+(float(ms[1])/1000000))
    av_time = np.mean(time_list)
    print(av_time)
    return

######################## SURROGATE TRAINING + BATCH SIM DATA QUALITY ########################
def graph_regressor_surrogate_scores(x):
    training_xp = []
    training_yp = []
    maximum_xp = [[]]
    maximum_yp = [[]]
    training_xs = []
    training_ys = []
    maximum_xs = [[]]
    maximum_ys = [[]]
    training_xt = []
    training_yt = []
    maximum_xt = [[]]
    maximum_yt = [[]]
    for j in range(5,24,6):
        strainp = []
        strains = []
        straint = []
        if os.path.exists(IN_PROGRESS+"/surrogate_model_training/"+str(x)+"/primary/surrogate_training_data"+str(j)+".csv"):
            prim = open(IN_PROGRESS+"/surrogate_model_training/"+str(x)+"/primary/surrogate_training_data"+str(j)+".csv","r")
            strainp = prim.readlines()
            prim.close()
        if os.path.exists(IN_PROGRESS+"/surrogate_model_training/"+str(x)+"/secondary/surrogate_training_data"+str(j)+".csv"):
            sec = open(IN_PROGRESS+"/surrogate_model_training/"+str(x)+"/secondary/surrogate_training_data"+str(j)+".csv","r")
            strains = sec.readlines()
            sec.close()
        if os.path.exists(IN_PROGRESS+"/surrogate_model_training/"+str(x)+"/tertiary/surrogate_training_data"+str(j)+".csv"):
            ter = open(IN_PROGRESS+"/surrogate_model_training/"+str(x)+"/tertiary/surrogate_training_data"+str(j)+".csv","r")
            straint = ter.readlines()
            ter.close()
        tempx = []
        tempy = []
        for i in strainp:
            st = i.rstrip().split(",")
            tempx.append(float(st[2]))
            tempy.append(float(st[3]))
            maximum_xp.append([float(st[2])])
            maximum_yp.append([float(st[5].split("]")[0])])
        training_xp.append(tempx)
        training_yp.append(tempy)
        tempx = []
        tempy = []
        for i in strains:
            st = i.rstrip().split(",")
            tempx.append(float(st[2]))
            tempy.append(float(st[3]))
            maximum_xs.append([float(st[2])])
            maximum_ys.append([float(st[5].split("]")[0])])
        training_xs.append(tempx)
        training_ys.append(tempy)
        tempx = []
        tempy = []
        for i in straint:
            st = i.rstrip().split(",")
            tempx.append(float(st[2]))
            tempy.append(float(st[3]))
            maximum_xt.append([float(st[2])])
            maximum_yt.append([float(st[5].split("]")[0])])
        training_xt.append(tempx)
        training_yt.append(tempy)
        tempx = []
        tempy = []
    training_xp = [x for x in training_xp if not len(x)==0]
    training_yp = [x for x in training_yp if not len(x)==0]
    training_xs = [x for x in training_xs if not len(x)==0]
    training_ys = [x for x in training_ys if not len(x)==0]
    training_xt = [x for x in training_xt if not len(x)==0]
    training_yt = [x for x in training_yt if not len(x)==0]
    txp = np.mean(training_xp,axis=0)
    typ = np.mean(training_yp,axis=0)
    txs = np.mean(training_xs,axis=0)
    tys = np.mean(training_ys,axis=0)
    txt = np.mean(training_xt,axis=0)
    tyt = np.mean(training_yt,axis=0)
    plt_lines([[txp,typ],[txs,tys],[txt,tyt]],"Training Data Size","Surrogate Score (sklearn.MLP.score)",["Primary Average","Secondary Average","Tertiary Average"],"Continuous Primary Fitness Surrogate Model prediction scores",IN_PROGRESS+"/graphs/multiple_surrogate_average_scores_"+str(x))
    return

def graph_classifier_surrogate_scores():
    maximum_x = []
    maximum_y = []
    loc = "/surrogate_model_training/primary/sampling_before/undersample/"
    strain = []
    for (a,b,files) in os.walk(IN_PROGRESS+loc):
        for file in files:
            if file.split("_")[0]=="surrogate":
                st = open(IN_PROGRESS+loc+file,"r")
                strain.append(st.readlines())
                st.close()
    ic = 0
    for j in strain:
        maximum_x.append([])
        maximum_y.append([])
        for i in j:
            st = i.rstrip().split(",")
            maximum_x[ic].append(float(st[2]))
            maximum_y[ic].append(float(st[5].split("]")[0]))
        ic+=1
    plt_classifier([maximum_x,maximum_y],"Training Data (data points)","Surrogate Score (sklearn.MLP.score)",["Model 1","Model 2","Model 3"],"Discrete Primary Fitness Surrogate Model prediction scores",IN_PROGRESS+"/graphs/multiple_surrogate_average_scores_primary_classifier")
    return

def plt_lines(data,xl,yl,l,title,save):
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=24, color='black')
    plt.xlabel(xl, fontsize=18)
    plt.ylabel(yl, fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylim(-1,1)
    count=0
    for i in data:
        plt.plot(i[0],i[1], label=l[count])
        count+=1
    plt.legend(frameon=True, fontsize=18, loc=3)
    plt.savefig(save+".png", bbox_inches='tight')
    plt.savefig(save+".pdf", bbox_inches='tight')
    plt.show()
    plt.close()
    return

def plt_classifier(data,xl,yl,l,title,save):
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=24, color='black')
    plt.xlabel(xl, fontsize=18)
    plt.ylabel(yl, fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylim(-1,1)
    count=0
    for i in range(len(data[0])):
        if any([x for x in data[0][i] if x>700]):
            plt.plot(data[0][i],data[1][i], label=l[count])
            count+=1
    plt.legend(frameon=True, fontsize=18, loc=3)
    plt.savefig(save+".png", bbox_inches='tight')
    plt.savefig(save+".pdf", bbox_inches='tight')
    plt.show()
    plt.close()
    return

def available_data_quality(cont_loc,disc_loc):
    bsd = open(disc_loc,"r")
    data = bsd.readlines()
    bsd.close()
    bsd2 = open(cont_loc,"r")
    data2 = bsd2.readlines()
    bsd2.close()
    fitnesses = []
    fitnesses2 = []
    for li in data:
        sp = li.rstrip().split(",")
        if sp[0]!="Seed" and sp[0]!="time_taken":
            fitnesses.append(apply_fitness(float(sp[10]),float(sp[11]),float(sp[12]))[0])
    for li in data2:
        sp = li.rstrip().split(",")
        if sp[0]!="Seed" and sp[0]!="time_taken":
            fitnesses2.append(apply_fitness2(float(sp[10]),float(sp[11]),float(sp[12]))[0])
    plt_hist([fitnesses,fitnesses2],["Discrete Primary Fitness","Continuous Primary Fitness"],"Potential Data Set Fitness Frequency",IN_PROGRESS+"/graphs/potential_data")
    return

def apply_fitness(prim,sec,ter):
    fitness = (1.0*prim)+(0.01*sec)-(0.00001*ter)
    return fitness,

def apply_fitness2(prim,sec,ter):
    fitness = (0.001*prim)+(0.01*sec)-(0.00001*ter)
    return fitness,

def plt_hist(data,labels,title,save):
    plt.figure(figsize=(12, 6))
    plt.ylim(-1,100)
    colours = ['b','y']
    for d in range(len(data)):
        n, bins, patches = plt.hist(x=data[d], bins='auto', alpha=0.65, rwidth=0.85, label=labels[d], color=colours[d])
    
    plt.xlabel("Fitness", fontsize=18)
    plt.ylabel("Frequency", fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(title, fontsize=24, color='black')
    plt.legend(frameon=True, fontsize=18, loc=2)
    plt.savefig(save+".png", bbox_inches='tight')
    plt.savefig(save+".pdf", bbox_inches='tight')
    plt.show()
    plt.close()
    return



######################## SURROGATE GA #####################################
def surrogate_ga_graphs(cont_loc,disc_loc):
    print("Graphing GA performance and variance.")
    data2,l2 = get_surrogate_data(0,disc_loc)
    data,l = get_surrogate_data(0,cont_loc)
    plt_std_simga([data[0],data2[0]],[data[1],data2[1]],[data[2],data2[2]],"Simulations Performed","Evaluated Fitness",["Continuous Primary Fitness","Discrete Primary Fitness"],"Increasing Fitness of "+str(l)+" Surrogate Model Guided ("+str(MU)+"+"+str(LAMBDA)+")GA",IN_PROGRESS+"/graphs/surrogate_ga_average")
    print("Graphs completed. Saved to "+IN_PROGRESS+".\n")
    return

def get_surrogate_data(sd,f):
    fitness_av = []
    fitness_std = []
    primary_av = []
    primary_std = []
    secondary_av = []
    secondary_std = []
    tertiary_av = []
    tertiary_std = []
    sims = []
    if not os.path.exists(IN_PROGRESS+f):
        print("please run multiple seeded GA before calling this function. seeded_ga_performances.csv must exist")
        exit(1)
    rf = open(IN_PROGRESS+f,"r")
    lcount = -1
    scount = sd
    wierd = 0
    #Check all GA's performance
    for li in rf:
        s = li.rstrip().replace(" ","").split("\t")
        #If not new GA
        if not s[0]=="gen":
            if not float(s[8])==-1 and wierd==0:
                fitness_av[lcount].append(float(s[2]))
                fitness_std[lcount].append(float(s[3]))
                primary_av[lcount].append(float(s[6]))
                primary_std[lcount].append(float(s[7]))
                secondary_av[lcount].append(float(s[10]))
                secondary_std[lcount].append(float(s[11]))
                tertiary_av[lcount].append(float(s[14]))
                tertiary_std[lcount].append(float(s[15]))
            else:
                wierd = 1
            if lcount==0:
                scount += int(s[1])
                sims.append(scount)
        else:
            if wierd==0:
                fitness_av.append([])
                fitness_std.append([])
                primary_av.append([])
                primary_std.append([])
                secondary_av.append([])
                secondary_std.append([])
                tertiary_av.append([])
                tertiary_std.append([])
                lcount += 1
            scount = sd
            wierd = 0
    data = len(primary_av)
    if not data/5==0 and data>5:
        data = (int(data/5))*5
        fitness_av = fitness_av[:data]
        fitness_std = fitness_std[:data]
        primary_av = primary_av[:data]
        primary_std = primary_std[:data]
        secondary_av = secondary_av[:data]
        secondary_std = secondary_std[:data]
        tertiary_av = tertiary_av[:data]
        tertiary_std = tertiary_std[:data]
    #Find average and average standard deviation at each of the GA generations for plotting
    fstd = fitness_std[0]
    pstd = primary_std[0]
    sstd = secondary_std[0]
    tstd = tertiary_std[0]
    for p in range(len(primary_av)):
        if any([x>=10 for x in primary_av[p]]):
            primary_av[p] = [float(x)/1000 for x in primary_av[p]]
            primary_std[p] = [float(x)/1000 for x in primary_std[p]]
    fav = [sum(x)/data for x in zip(*fitness_av)]
    zippedf = [x for x in zip(*fitness_std)]
    if len(zippedf[0])>1:
        fstd = np.std(np.array(zippedf),axis=1)
    pav = [sum(x)/data for x in zip(*primary_av)]
    zippedp = [x for x in zip(*primary_std)]
    if len(zippedp[0])>1:
        pstd = np.std(np.array(zippedp),axis=1)
    sav = [sum(x)/data for x in zip(*secondary_av)]#0.01*
    zippeds = [x for x in zip(*secondary_std)]
    if len(zippeds[0])>1:
        sstd = np.std(np.array(zippeds),axis=1)
    tav = [sum(x)/data for x in zip(*tertiary_av)]#0.00001*
    zippedt = [x for x in zip(*tertiary_std)]
    if len(zippedt[0])>1:
        tstd = np.std(np.array(zippedt),axis=1)
    return [sims,fav,fstd,pav,pstd,sav,sstd,tav,tstd],data

def plt_std_surrogate(x,y,std,xl,yl,l,title,save):
    plt.figure(figsize=(15, 6))
    plt.title(title, fontsize=24, color='black')
    plt.xlabel(xl, fontsize=18)
    plt.ylabel(yl, fontsize=18)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.errorbar(x,y, yerr=std, fmt='b-', ecolor='r', label=l, elinewidth=1, capsize=2)
    plt.legend(frameon=True, fontsize=18, loc=2)
    plt.tight_layout()
    plt.savefig(save+"_std.png", bbox_inches='tight')
    plt.savefig(save+"_std.pdf", bbox_inches='tight')
    plt.show()
    plt.close()
    return

###################################### SURRSIM COMPARISON #################################    
def plt_std_surrsim(x,y,std,xl,yl,l,title,save):
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=24, color='black')
    plt.xlabel(xl, fontsize=14)
    plt.ylabel(yl, fontsize=14)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    linest = ['-','--']
    elinst = ['--',':']
    for i in range(len(x)):
        for line in range(len(y)):
            eb = plt.errorbar(x[i],y[i][line], yerr=std[i][line], label=l[i*len(x)+line], ls=linest[i], elinewidth=1, capsize=2)
            eb[-1][0].set_linestyle(elinst[line])
    plt.legend(frameon=True, fontsize=18, loc=4)
    plt.savefig(save+"_std.png", bbox_inches='tight')
    plt.savefig(save+"_std.pdf", bbox_inches='tight')
    plt.show()
    plt.close()
    return

def surrogate_ga_sim_eval_graphs(cont_loc,disc_loc):
    print("Graphing GA performance and variance.",cont_loc)
    data,sim_data,l,mu,lam = get_surrsim_data(0,cont_loc)
    data2,sim_data2,l2,mu2,lam2 = get_surrsim_data(0,disc_loc)
    check = [data[1],sim_data[1],data[2],sim_data[2],data[3],sim_data[3],data[4],sim_data[4],data[5],sim_data[5],data[6],sim_data[6],data[7],sim_data[7],data[8],sim_data[8]]
    if any([len(x)<len(data[0]) for x in check]):
        data[0] = data[0][:min([len(x) for x in check])]
    if any([len(x)<len(sim_data[0]) for x in check]):
        sim_data[0] = sim_data[0][:min([len(x) for x in check])]
    check2 = [data2[1],sim_data2[1],data2[2],sim_data2[2],data2[3],sim_data2[3],data2[4],sim_data2[4],data2[5],sim_data2[5],data2[6],sim_data2[6],data2[7],sim_data2[7],data2[8],sim_data2[8]]
    if any([len(x)<len(data2[0]) for x in check2]):
        data2[0] = data2[0][:min([len(x) for x in check2])]
    if any([len(x)<len(sim_data2[0]) for x in check2]):
        sim_data2[0] = sim_data2[0][:min([len(x) for x in check2])]
    plt_std_surrsim([data[0],data2[0]],[[data[1],sim_data[1]],[data2[1],sim_data2[1]]],[[data[2],sim_data[2]],[data2[2],sim_data2[2]]],"Solutions Evaluated","Total Fitness",["Continuous Primary - Surrogate Predicted","Continuous Primary - Simulation Evaluated","Discrete Primary - Surrogate Predicted","Discrete Primary - Simulation Evaluated"],"Surrogate GA with Simulation Evaluation of Population per Generation",IN_PROGRESS+"/graphs/surrsim_ga_average")
#    plt_std_surrsim([data2[0]],[data2[1],sim_data2[1]],[data2[2],sim_data2[2]],"Solutions Evaluated","Total Fitness",["Continuous Primary - Surrogate Predicted","Continuous Primary - Simulation Evaluated","Discrete Primary - Surrogate Predicted","Discrete Primary - Simulation Evaluated"],"Surrogate GA with Simulation Evaluation of Population per Generation",IN_PROGRESS+"/graphs/surrsim_ga_average")
    print("Graphs completed. Saved to "+IN_PROGRESS+".\n")
    return

def get_surrsim_data(sd,f):
    fitness_av_surr = []
    fitness_std_surr = []
    primary_av_surr = []
    primary_std_surr = []
    secondary_av_surr = []
    secondary_std_surr = []
    tertiary_av_surr = []
    tertiary_std_surr = []
    fitness_av_sim = []
    fitness_std_sim = []
    primary_av_sim = []
    primary_std_sim = []
    secondary_av_sim = []
    secondary_std_sim = []
    tertiary_av_sim = []
    tertiary_std_sim = []
    sims = []
    if not os.path.exists(IN_PROGRESS+f+"_surrogate_ga_performances.csv"):
        print("please run multiple seeded GA before calling this function. seeded_ga_performances.csv must exist")
        exit(1)
    if not os.path.exists(IN_PROGRESS+f+"_surrogate_ga_simulation_evaluation.csv"):
        print("please run multiple seeded GA before calling this function. seeded_ga_performances.csv must exist")
        exit(1)
    surrf = open(IN_PROGRESS+f+"_surrogate_ga_performances.csv","r")
    data = [[],[]]
    data[0] = surrf.readlines()
    surrf.close()
    simf = open(IN_PROGRESS+f+"_surrogate_ga_simulation_evaluation.csv","r")
    data[1] = simf.readlines()
    simf.close()
    lcount = -1
    lcount2 = -1
    mcount = 0
    m = 0
    l = 0
    scount = sd
    tcount = 0
    #Check all GA's performance
    for typ in data:
        for li in typ:
            s = li.rstrip().replace(" ","").split("\t")
            #If not new GA
            if not s[0]=="gen":
                if mcount==0:
                    m = int(s[1])
                    mcount+=1
                elif mcount==1:
                    l = int(s[1])
                    mcount+=1
                if tcount==0:
                    if not (float(s[8])==-1000 or float(s[8])==-1):
                        fitness_av_surr[lcount].append(float(s[2]))
                        fitness_std_surr[lcount].append(float(s[3]))
                        primary_av_surr[lcount].append(float(s[6]))
                        primary_std_surr[lcount].append(float(s[7]))
                        secondary_av_surr[lcount].append(float(s[10]))
                        secondary_std_surr[lcount].append(float(s[11]))
                        tertiary_av_surr[lcount].append(float(s[14]))
                        tertiary_std_surr[lcount].append(float(s[15]))
                    if not len(sims)==GENERATIONS+1:
                        scount += int(s[1])
                        sims.append(scount)
                else:
                    if not (float(s[8])==-1000 or float(s[8])==-1):
                        fitness_av_sim[lcount2].append(float(s[2]))
                        fitness_std_sim[lcount2].append(float(s[3]))
                        primary_av_sim[lcount2].append(float(s[6]))
                        primary_std_sim[lcount2].append(float(s[7]))
                        secondary_av_sim[lcount2].append(float(s[10]))
                        secondary_std_sim[lcount2].append(float(s[11]))
                        tertiary_av_sim[lcount2].append(float(s[14]))
                        tertiary_std_sim[lcount2].append(float(s[15]))
            else:
                if tcount==1:
                    fitness_av_sim.append([])
                    fitness_std_sim.append([])
                    primary_av_sim.append([])
                    primary_std_sim.append([])
                    secondary_av_sim.append([])
                    secondary_std_sim.append([])
                    tertiary_av_sim.append([])
                    tertiary_std_sim.append([])
                    lcount2 += 1
                else:
                    fitness_av_surr.append([])
                    fitness_std_surr.append([])
                    primary_av_surr.append([])
                    primary_std_surr.append([])
                    secondary_av_surr.append([])
                    secondary_std_surr.append([])
                    tertiary_av_surr.append([])
                    tertiary_std_surr.append([])
                    lcount += 1
                if not len(sims)==GENERATIONS+1:
                    sims = []
                scount = sd
        tcount += 1
    print("f",len(fitness_av_surr))
    fitness_av_surr = [x for x in fitness_av_surr if len(x)==GENERATIONS+1]
    for index in range(len(fitness_av_surr[0])):
        testerino = [x[index] for x in fitness_av_surr]
        if -10 in testerino:
            for i in fitness_av_surr:
                i[index] = -0.1
                
    fitness_std_surr = [x for x in fitness_std_surr if len(x)==GENERATIONS+1]
    for index in range(len(fitness_std_surr[0])):
        testerino = [x[index] for x in fitness_std_surr]
        if 0 in testerino:
            for i in fitness_std_surr:
                i[index] = 0
                
    primary_av_surr = [x for x in primary_av_surr if len(x)==GENERATIONS+1]
    for index in range(len(primary_av_surr[0])):
        testerino = [x[index] for x in primary_av_surr]
        if -1000 in testerino:
            for i in primary_av_surr:
                i[index] = -10
                
    primary_std_surr = [x for x in primary_std_surr if len(x)==GENERATIONS+1]
    for index in range(len(primary_std_surr[0])):
        testerino = [x[index] for x in primary_std_surr]
        if 0 in testerino:
            for i in primary_std_surr:
                i[index] = 0
                
    secondary_av_surr = [x for x in secondary_av_surr if len(x)==GENERATIONS+1]
    for index in range(len(secondary_av_surr[0])):
        testerino = [x[index] for x in secondary_av_surr]
        if -100 in testerino:
            for i in secondary_av_surr:
                i[index] = -1
            
    secondary_std_surr = [x for x in secondary_std_surr if len(x)==GENERATIONS+1]
    for index in range(len(secondary_std_surr[0])):
        testerino = [x[index] for x in secondary_std_surr]
        if 0 in testerino:
            for i in secondary_std_surr:
                i[index] = 0
                
    tertiary_av_surr = [x for x in tertiary_av_surr if len(x)==GENERATIONS+1]
    for index in range(len(tertiary_av_surr[0])):
        testerino = [x[index] for x in tertiary_av_surr]
        if 100000 in testerino:
            for i in tertiary_av_surr:
                i[index] = -100
                
    tertiary_std_surr = [x for x in tertiary_std_surr if len(x)==GENERATIONS+1]
    for index in range(len(tertiary_std_surr[0])):
        testerino = [x[index] for x in tertiary_std_surr]
        if 0 in testerino:
            for i in tertiary_std_surr:
                i[index] = 0
                
    fitness_av_sim = [x for x in fitness_av_sim if len(x)==GENERATIONS+1]
    for index in range(len(fitness_av_sim[0])):
        testerino = [x[index] for x in fitness_av_sim]
        if -10 in testerino:
            for i in fitness_av_sim:
                i[index] = -0.1
                
    fitness_std_sim = [x for x in fitness_std_sim if len(x)==GENERATIONS+1]
    for index in range(len(fitness_std_sim[0])):
        testerino = [x[index] for x in fitness_std_sim]
        if 0 in testerino:
            for i in fitness_std_sim:
                i[index] = 0
                
    primary_av_sim = [x for x in primary_av_sim if len(x)==GENERATIONS+1]
    for index in range(len(primary_av_sim[0])):
        testerino = [x[index] for x in primary_av_sim]
        if -1000 in testerino:
            for i in primary_av_sim:
                i[index] = -10
                
    primary_std_sim = [x for x in primary_std_sim if len(x)==GENERATIONS+1]
    for index in range(len(primary_std_sim[0])):
        testerino = [x[index] for x in primary_std_sim]
        if 0 in testerino:
            for i in primary_std_sim:
                i[index] = 0
                
    secondary_av_sim = [x for x in secondary_av_sim if len(x)==GENERATIONS+1]
    for index in range(len(secondary_av_sim[0])):
        testerino = [x[index] for x in secondary_av_sim]
        if -100 in testerino:
            for i in secondary_av_sim:
                i[index] = -1
            
    secondary_std_sim = [x for x in secondary_std_sim if len(x)==GENERATIONS+1]
    for index in range(len(secondary_std_sim[0])):
        testerino = [x[index] for x in secondary_std_sim]
        if 0 in testerino:
            for i in secondary_std_sim:
                i[index] = 0
                
    tertiary_av_sim = [x for x in tertiary_av_sim if len(x)==GENERATIONS+1]
    for index in range(len(tertiary_av_sim[0])):
        testerino = [x[index] for x in tertiary_av_sim]
        if 100000 in testerino:
            for i in tertiary_av_sim:
                i[index] = -100
                
    tertiary_std_sim = [x for x in tertiary_std_sim if len(x)==GENERATIONS+1]
    for index in range(len(tertiary_std_sim[0])):
        testerino = [x[index] for x in tertiary_std_sim]
        if 0 in testerino:
            for i in tertiary_std_sim:
                i[index] = 0
    data = len(primary_av_surr)
    #Find average and average standard deviation at each of the GA generations for plotting
    if len(fitness_std_surr)>0:
        fstd = fitness_std_surr[0]
        fav = [sum(x)/len(fitness_av_surr) for x in zip(*fitness_av_surr)]
        zippedf = [x for x in zip(*fitness_std_surr)]
        if len(zippedf[0])>1:
            fstd = np.std(np.array(zippedf),axis=1)
    if len(primary_std_surr)>0:
        pstd = primary_std_surr[0]
        pav = [sum(x)/len(primary_av_surr) for x in zip(*primary_av_surr)]
        zippedp = [x for x in zip(*primary_std_surr)]
        if len(zippedp[0])>1:
            pstd = np.std(np.array(zippedp),axis=1)
    if len(secondary_std_surr)>0:
        sstd = secondary_std_surr[0]
        sav = [sum(x)/len(secondary_av_surr) for x in zip(*secondary_av_surr)]#0.01*
        zippeds = [x for x in zip(*secondary_std_surr)]
        if len(zippeds[0])>1:
            sstd = np.std(np.array(zippeds),axis=1)
    if len(tertiary_std_surr)>0:
        tstd = tertiary_std_surr[0]
        tav = [sum(x)/len(tertiary_av_surr) for x in zip(*tertiary_av_surr)]#0.00001*
        zippedt = [x for x in zip(*tertiary_std_surr)]
        if len(zippedt[0])>1:
            tstd = np.std(np.array(zippedt),axis=1)
    #Find average and average standard deviation at each of the GA generations for plotting
    if len(fitness_std_sim)>0:
        fstd2 = fitness_std_sim[0]
        fav2 = [sum(x)/len(fitness_av_sim) for x in zip(*fitness_av_sim)]
        zippedf2 = [x for x in zip(*fitness_std_sim)]
        if len(zippedf2[0])>1:
            fstd2 = np.std(np.array(zippedf2),axis=1)
    if len(primary_std_sim)>0:
        pstd2 = primary_std_sim[0]
        pav2 = [sum(x)/len(primary_av_sim) for x in zip(*primary_av_sim)]
        zippedp2 = [x for x in zip(*primary_std_sim)]
        if len(zippedp2[0])>1:
            pstd2 = np.std(np.array(zippedp2),axis=1)
    if len(secondary_std_sim)>0:
        sstd2 = secondary_std_sim[0]
        sav2 = [sum(x)/len(secondary_av_sim) for x in zip(*secondary_av_sim)]#0.01*
        zippeds2 = [x for x in zip(*secondary_std_sim)]
        if len(zippeds2[0])>1:
            sstd2 = np.std(np.array(zippeds2),axis=1)
    if len(tertiary_std_sim)>0:
        tstd2 = tertiary_std_sim[0]
        tav2 = [sum(x)/len(tertiary_av_sim) for x in zip(*tertiary_av_sim)]#0.00001*
        zippedt2 = [x for x in zip(*tertiary_std_sim)]
        if len(zippedt2[0])>1:
            tstd2 = np.std(np.array(zippedt2),axis=1)
    return [sims,fav,fstd,pav,pstd,sav,sstd,tav,tstd],[sims,fav2,fstd2,pav2,pstd2,sav2,sstd2,tav2,tstd2],data,m,l



################################# GPU TIME BENCHMARK ##################################
def gpu_benchmark_graph():
    r = open(IN_PROGRESS+"../parameter-search\PredPrey\paper_graphs\performance_graph/results.csv","r")
    res = [[],[]]
    for l in r:
        sp = l.split(",")
        if not(sp[0]=="num_parameters"):
            res[0].append(int(sp[1]))
            res[1].append(float(sp[3]))
    title = "Batch Simulation Performance over multiple available GPUs"
    plt.figure(figsize=(12,4))
    plt.title(title, fontsize=24, color='black')
    plt.plot(res[0], res[1], label="10k simulations using all available devices")
    plt.xlabel("GPUs available", fontsize=18)
    plt.ylabel("Time to full completion(seconds)", fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks([1,2,3,4])
    plt.legend(frameon=True, fontsize=18, loc=1)
    plt.xticks(fontsize=12)
    plt.savefig(IN_PROGRESS+"/gpu_performance_graph.png", bbox_inches='tight')
    plt.savefig(IN_PROGRESS+"/gpu_performance_graph.pdf", bbox_inches='tight')
    return


###################################### GENERATE GRAPHS ###############################
training_datasets = [5000,10000,15000,25000,35000,50000]
GENERATIONS = 196
IN_PROGRESS = os.getcwd()+"/"
MU = 100
LAMBDA = 25
if not os.path.exists(IN_PROGRESS+"/graphs/"):
    os.mkdir(IN_PROGRESS+"/graphs/")
#Sim Ga Baseline
simulation_ga_graphs("/simulation_ga/continuous_fitness/continuous_sim_ga_performances.csv","/simulation_ga/discrete_fitness/discrete_sim_ga_performances.csv")
#Batch Simulation Data Quality
available_data_quality(IN_PROGRESS+"/batch_simulation_data/continuous/batch_simulation_data.csv",IN_PROGRESS+"/batch_simulation_data/discrete/batch_simulation_data.csv")
#Surrogate model training graphs
graph_regressor_surrogate_scores(50000)
graph_classifier_surrogate_scores()
#Surrogate GAs
surrogate_ga_graphs("/surrogate_ga/continuous_fitness/50000/continuous_surrogate_ga_performances.csv","/surrogate_ga/discrete_fitness/50000/discrete_surrogate_ga_performances.csv")
#Surrogate GA with sim comparison per generation
surrogate_ga_sim_eval_graphs("/surrogate_vs_sim_comparison/continuous_fitness/50000/continuous","/surrogate_vs_sim_comparison/discrete_fitness/50000/discrete")