# 6.00.2x Problem Set 4

import numpy
import random
import pylab
from ps3b import *


# PROBLEM 1
#
def simulationDelayedTreatment(numTrials):
    """
    Runs simulations and make histograms for problem 1.

    Runs numTrials simulations to show the relationship between delayed
    treatment and patient outcome using a histogram.

    Histograms of final total virus populations are displayed for delays of 300,
    150, 75, 0 timesteps (followed by an additional 150 timesteps of
    simulation).

    numTrials: number of simulation runs to execute (an integer)
    """

    # TODO
    delayTimes  = [300, 150, 75, 0]
    index = 0
    fig, ax = pylab.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for delayTime in delayTimes:
        numTotalPop = simulationWith1DrugDelay(100, 1000, 0.1, 0.05, {'guttagonol': False}, 0.005, numTrials, delayTime)
        print "the delay time ", delayTime, numTotalPop
        index += 1
        pylab.subplot(2, 2, index)
        pylab.hist( numTotalPop,  bins=range(min(numTotalPop), max(numTotalPop)+50, 50) )
        pylab.xlabel( "delay = {}".format(delayTime) )
    fig.text(0.5, 0.04, '#Virus', ha='center')
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
    fig.text(0.5, 1.1, "Treated Patient simulation", ha='center')
    pylab.show()

def simulationWith1DrugDelay(numViruses, maxPop, maxBirthProb, clearProb, resistances,
                       mutProb, numTrials, delayTime):
    """
    Runs simulations and plots graphs for problem 5.

    For each of numTrials trials, instantiates a patient, runs a simulation for
    150 timesteps, adds guttagonol, and runs the simulation for an additional
    150 timesteps.  At the end plots the average virus population size
    (for both the total virus population and the guttagonol-resistant virus
    population) as a function of time.

    numViruses: number of ResistantVirus to create for patient (an integer)
    maxPop: maximum virus population for patient (an integer)
    maxBirthProb: Maximum reproduction probability (a float between 0-1)
    clearProb: maximum clearance probability (a float between 0-1)
    resistances: a dictionary of drugs that each ResistantVirus is resistant to
                 (e.g., {'guttagonol': False})
    mutProb: mutation probability for each ResistantVirus particle
             (a float between 0-1).
    numTrials: number of simulation runs to execute (an integer)
    delayTime: delayed timesteps to add the drugs
    """

    # TODO
    timesteps = 150 + delayTime
    numTotalPop = [  ]

    for i in range(numTrials):
        viruses = [ ResistantVirus(maxBirthProb , clearProb, resistances.copy(), mutProb) for ii in range(numViruses)]
        patient = TreatedPatient(viruses, maxPop)
        # print "simulationWithDrugDelay(numViruses, maxPop, maxBirthProb, clearProb, resistances, mutProb, numTrials, delayTime)"
        # print numViruses, maxPop, maxBirthProb, clearProb, resistances, mutProb, numTrials, delayTime
        for j in range(timesteps):
            if j == delayTime:
                patient.addPrescription("guttagonol")
            patient.update()
        # print len(patient.viruses)
        numTotalPop.append( len(patient.viruses) )
    return numTotalPop



#
# PROBLEM 2
#
def simulationTwoDrugsDelayedTreatment(numTrials):
    """
    Runs simulations and make histograms for problem 2.

    Runs numTrials simulations to show the relationship between administration
    of multiple drugs and patient outcome.

    Histograms of final total virus populations are displayed for lag times of
    300, 150, 75, 0 timesteps between adding drugs (followed by an additional
    150 timesteps of simulation).

    numTrials: number of simulation runs to execute (an integer)
    """
    # TODO
    delayTimes  = [300, 150, 75, 0]
    index = 0
    fig, ax = pylab.subplots(nrows=2, ncols=2, figsize=(6, 6))
    for delayTime in delayTimes:
        numTotalPop = simulationWith2DrugsDelay(100, 1000, 0.1, 0.05, {'guttagonol': False, 'grimpex': False}, 0.005, numTrials, delayTime)
        print "the delay time ", delayTime, numTotalPop
        index += 1
        pylab.subplot(2, 2, index)
        pylab.hist( numTotalPop,  bins=range(min(numTotalPop), max(numTotalPop)+50, 50) )
        pylab.xlabel( "delay = {}".format(delayTime) )
    fig.text(0.5, 0.04, '#Virus', ha='center')
    fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical')
    fig.text(0.5, 1.1, "Treated Patient simulation", ha='center')
    pylab.show()


def simulationWith2DrugsDelay(numViruses, maxPop, maxBirthProb, clearProb, resistances,
                       mutProb, numTrials, delayTime):
    """
    Runs simulations and plots graphs for problem 5.

    For each of numTrials trials, instantiates a patient, runs a simulation for
    150 timesteps, adds guttagonol, and runs the simulation for an additional
    150 timesteps.  At the end plots the average virus population size
    (for both the total virus population and the guttagonol-resistant virus
    population) as a function of time.

    numViruses: number of ResistantVirus to create for patient (an integer)
    maxPop: maximum virus population for patient (an integer)
    maxBirthProb: Maximum reproduction probability (a float between 0-1)
    clearProb: maximum clearance probability (a float between 0-1)
    resistances: a dictionary of drugs that each ResistantVirus is resistant to
                 (e.g., {'guttagonol': False})
    mutProb: mutation probability for each ResistantVirus particle
             (a float between 0-1).
    numTrials: number of simulation runs to execute (an integer)
    delayTime: delayed timesteps to add the drugs
    """

    # TODO
    timesteps = 150 + delayTime + 150
    numTotalPop = [  ]

    for i in range(numTrials):
        viruses = [ ResistantVirus(maxBirthProb , clearProb, resistances.copy(), mutProb) for ii in range(numViruses)]
        patient = TreatedPatient(viruses, maxPop)
        # print "simulationWithDrugDelay(numViruses, maxPop, maxBirthProb, clearProb, resistances, mutProb, numTrials, delayTime)"
        # print numViruses, maxPop, maxBirthProb, clearProb, resistances, mutProb, numTrials, delayTime
        for j in range(timesteps):
            if j == 150:
                patient.addPrescription("guttagonol")
            if j == 150 + delayTime:
                patient.addPrescription("grimpex")
            patient.update()
        # print len(patient.viruses)
        numTotalPop.append( len(patient.viruses) )
    return numTotalPop

# simulationTwoDrugsDelayedTreatment(100)
simulationDelayedTreatment(100)