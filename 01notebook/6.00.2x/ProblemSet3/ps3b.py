# Problem Set 3: Simulating the Spread of Disease and Virus Population Dynamics

import numpy
import random
import pylab

'''
Begin helper code
'''

class NoChildException(Exception):
    """
    NoChildException is raised by the reproduce() method in the SimpleVirus
    and ResistantVirus classes to indicate that a virus particle does not
    reproduce. You can use NoChildException as is, you do not need to
    modify/add any code.
    """

'''
End helper code
'''

#
# PROBLEM 2
#
class SimpleVirus(object):

    """
    Representation of a simple virus (does not model drug effects/resistance).
    """
    def __init__(self, maxBirthProb, clearProb):
        """
        Initialize a SimpleVirus instance, saves all parameters as attributes
        of the instance.
        maxBirthProb: Maximum reproduction probability (a float between 0-1)
        clearProb: Maximum clearance probability (a float between 0-1).
        """
        # TODO
        self.maxBirthProb = maxBirthProb
        self.clearProb = clearProb


    def getMaxBirthProb(self):
        """
        Returns the max birth probability.
        """
        # TODO
        return self.maxBirthProb

    def getClearProb(self):
        """
        Returns the clear probability.
        """
        # TODO
        return self.clearProb

    def doesClear(self):
        """ Stochastically determines whether this virus particle is cleared from the
        patient's body at a time step.
        returns: True with probability self.getClearProb and otherwise returns
        False.
        """

        # TODO
        theProb = random.random()
        clearProb = self.getClearProb()
        if theProb < clearProb : ## clearProb = 0, theProb = 0, should return false, not clear
            return True
        else:
            return False

    def reproduce(self, popDensity):
        """
        Stochastically determines whether this virus particle reproduces at a
        time step. Called by the update() method in the Patient and
        TreatedPatient classes. The virus particle reproduces with probability
        self.maxBirthProb * (1 - popDensity).

        If this virus particle reproduces, then reproduce() creates and returns
        the instance of the offspring SimpleVirus (which has the same
        maxBirthProb and clearProb values as its parent).

        popDensity: the population density (a float), defined as the current
        virus population divided by the maximum population.

        returns: a new instance of the SimpleVirus class representing the
        offspring of this virus particle. The child should have the same
        maxBirthProb and clearProb values as this virus. Raises a
        NoChildException if this virus particle does not reproduce.
        """

        # TODO
        theProb = random.random()
        reproduceProb = self.getMaxBirthProb()*(1 - popDensity)
        if theProb < reproduceProb: # when popDensity > 1, reproduceProb<0, stop fertilize
            return SimpleVirus(self.getMaxBirthProb(), self.getClearProb())
        else:
            raise NoChildException()

class Patient(object):
    """
    Representation of a simplified patient. The patient does not take any drugs
    and his/her virus populations have no drug resistance.
    """

    def __init__(self, viruses, maxPop):
        """
        Initialization function, saves the viruses and maxPop parameters as
        attributes.

        viruses: the list representing the virus population (a list of
        SimpleVirus instances)

        maxPop: the maximum virus population for this patient (an integer)
        """

        # TODO
        self.viruses = viruses
        self.maxPop = maxPop

    def getViruses(self):
        """
        Returns the viruses in this Patient.
        """
        # TODO
        return self.viruses

    def getMaxPop(self):
        """
        Returns the max population.
        """
        # TODO
        return self.maxPop


    def getTotalPop(self):
        """
        Gets the size of the current total virus population.
        returns: The total virus population (an integer)
        """

        # TODO
        return len(self.viruses)

    def update(self):
        """
        Update the state of the virus population in this patient for a single
        time step. update() should execute the following steps in this order:

        - Determine whether each virus particle survives and updates the list
        of virus particles accordingly.

        - The current population density is calculated. This population density
          value is used until the next call to update()

        - Based on this value of population density, determine whether each
          virus particle should reproduce and add offspring virus particles to
          the list of viruses in this patient.

        returns: The total virus population at the end of the update (an
        integer)
        """

        # TODO
        # for i in range( len(self.getViruses())-1, -1, -1 ):
        #     curVirus = self.viruses[i]
        #     if curVirus.doesClear():
        #         del self.viruses[i] ## self.viruses.remove(curVirus)
        for curVirus in self.getViruses():
            if curVirus.doesClear():
                self.viruses.remove(curVirus) ## self.viruses.remove(curVirus)

        popDensity = 1.0 * self.getTotalPop() / self.getMaxPop()

        for i in range( len(self.getViruses()) ):
            curVirus = self.viruses[i]
            # childVirus =  curVirus.reproduce(popDensity)
            # if childVirus != None:
            #     (self.viruses).append(childVirus)
            try:
                self.viruses.append(curVirus.reproduce(popDensity))
            except NoChildException:
                pass

        return len(self.viruses)


# Viruses = [SimpleVirus(1.0, 0.0) for i in range(1)]
# P = Patient(Viruses, 100)
# for i in range(100):
#     P.update()
# print len(P.viruses)
#for virus in P.viruses:
#    print virus.maxBirthProb, virus.clearProb

#
# PROBLEM 3
#
def simulationWithoutDrug(numViruses, maxPop, maxBirthProb, clearProb,
                          numTrials):
    """
    Run the simulation and plot the graph for problem 3 (no drugs are used,
    viruses do not have any drug resistance).
    For each of numTrials trial, instantiates a patient, runs a simulation
    for 300 timesteps, and plots the average virus population size as a
    function of time.

    numViruses: number of SimpleVirus to create for patient (an integer)
    maxPop: maximum virus population for patient (an integer)
    maxBirthProb: Maximum reproduction probability (a float between 0-1)
    clearProb: Maximum clearance probability (a float between 0-1)
    numTrials: number of simulation runs to execute (an integer)
    """

    # TODO
    avgNumVirusList = []
    timesteps = 300
    for i in range(numTrials):
        viruses = [ SimpleVirus(maxBirthProb, clearProb) for ii in range(numViruses)]
        patient = Patient(viruses, maxPop)
        for j in range(timesteps):
            if i == 0 :
                avgNumVirusList.append(0.)
            avgNumVirusList[j] += patient.update()
    for i in range(timesteps):
        avgNumVirusList[i] /= numTrials

    # print avgNumVirusList
    fig = pylab.plot( avgNumVirusList )
    pylab.xlabel("time")
    pylab.ylabel("numViruses")
    pylab.title("SimpleVirus simulation")
    pylab.legend()
    pylab.show()

## test problem 3
# simulationWithoutDrug( 10, 1000, 0.6, 0.2, 5)
# simulationWithoutDrug(1, 10, 1.0, 0.0, 1)


#
# PROBLEM 4
#
class ResistantVirus(SimpleVirus):
    """
    Representation of a virus which can have drug resistance.
    """

    def __init__(self, maxBirthProb, clearProb, resistances, mutProb):
        """
        Initialize a ResistantVirus instance, saves all parameters as attributes
        of the instance.

        maxBirthProb: Maximum reproduction probability (a float between 0-1)

        clearProb: Maximum clearance probability (a float between 0-1).

        resistances: A dictionary of drug names (strings) mapping to the state
        of this virus particle's resistance (either True or False) to each drug.
        e.g. {'guttagonol':False, 'srinol':False}, means that this virus
        particle is resistant to neither guttagonol nor srinol.

        mutProb: Mutation probability for this virus particle (a float). This is
        the probability of the offspring acquiring or losing resistance to a drug.
        """

        # TODO
        # self.maxBirthProb = maxBirthProb
        # self.clearProb = clearProb
        # ypc 1: parent class init
        SimpleVirus.__init__(self, maxBirthProb, clearProb)
        self.resistances = resistances
        self.mutProb = mutProb

    def getResistances(self):
        """
        Returns the resistances for this virus.
        """
        # TODO
        return self.resistances

    def getMutProb(self):
        """
        Returns the mutation probability for this virus.
        """
        # TODO
        return self.mutProb

    def isResistantTo(self, drug):
        """
        Get the state of this virus particle's resistance to a drug. This method
        is called by getResistPop() in TreatedPatient to determine how many virus
        particles have resistance to a drug.

        drug: The drug (a string)

        returns: True if this virus instance is c to the drug, False
        otherwise.
        """

        # TODO
        keys = self.resistances.keys()
        if drug in keys:
            return self.resistances[drug]
        else:
            return False


    def reproduce(self, popDensity, activeDrugs):
        """
        Stochastically determines whether this virus particle reproduces at a
        time step. Called by the update() method in the TreatedPatient class.

        A virus particle will only reproduce if it is resistant to ALL the drugs
        in the activeDrugs list. For example, if there are 2 drugs in the
        activeDrugs list, and the virus particle is resistant to 1 or no drugs,
        then it will NOT reproduce.

        Hence, if the virus is resistant to all drugs
        in activeDrugs, then the virus reproduces with probability:

        self.maxBirthProb * (1 - popDensity).

        If this virus particle reproduces, then reproduce() creates and returns
        the instance of the offspring ResistantVirus (which has the same
        maxBirthProb and clearProb values as its parent). The offspring virus
        will have the same maxBirthProb, clearProb, and mutProb as the parent.

        For each drug resistance trait of the virus (i.e. each key of
        self.resistances), the offspring has probability 1-mutProb of
        inheriting that resistance trait from the parent, and probability
        mutProb of switching that resistance trait in the offspring.

        For example, if a virus particle is resistant to guttagonol but not
        srinol, and self.mutProb is 0.1, then there is a 10% chance that
        that the offspring will lose resistance to guttagonol and a 90%
        chance that the offspring will be resistant to guttagonol.
        There is also a 10% chance that the offspring will gain resistance to
        srinol and a 90% chance that the offspring will not be resistant to
        srinol.

        popDensity: the population density (a float), defined as the current
        virus population divided by the maximum population

        activeDrugs: a list of the drug names acting on this virus particle
        (a list of strings).

        returns: a new instance of the ResistantVirus class representing the
        offspring of this virus particle. The child should have the same
        maxBirthProb and clearProb values as this virus. Raises a
        NoChildException if this virus particle does not reproduce.
        """

        # TODO
        notReproduce = False
        for drug in activeDrugs:
            if not self.isResistantTo(drug):
                notReproduce = True
                break

        if False == notReproduce and random.random() < self.maxBirthProb * (1 - popDensity):
            for key, val in self.resistances.items():
                if random.random() < self.mutProb:
                    self.resistances[key] = (not val)
            return ResistantVirus(self.maxBirthProb, self.clearProb,  self.resistances, self.mutProb)
        else:
            raise NoChildException


class TreatedPatient(Patient):
    """
    Representation of a patient. The patient is able to take drugs and his/her
    virus population can acquire resistance to the drugs he/she takes.
    """

    def __init__(self, viruses, maxPop):
        """
        Initialization function, saves the viruses and maxPop parameters as
        attributes. Also initializes the list of drugs being administered
        (which should initially include no drugs).

        viruses: The list representing the virus population (a list of
        virus instances)

        maxPop: The  maximum virus population for this patient (an integer)
        """

        # TODO
        Patient.__init__(self, viruses, maxPop)
        # self.viruses = viruses
        # self.maxPop = maxPop
        self.activeDrugs = []

    def addPrescription(self, newDrug):
        """
        Administer a drug to this patient. After a prescription is added, the
        drug acts on the virus population for all subsequent time steps. If the
        newDrug is already prescribed to this patient, the method has no effect.

        newDrug: The name of the drug to administer to the patient (a string).

        postcondition: The list of drugs being administered to a patient is updated
        """

        # TODO
        if newDrug not in self.activeDrugs:
            self.activeDrugs.append(newDrug)


    def getPrescriptions(self):
        """
        Returns the drugs that are being administered to this patient.

        returns: The list of drug names (strings) being administered to this
        patient.
        """

        # TODO
        return self.activeDrugs


    def getResistPop(self, drugResist):
        """
        Get the population of virus particles resistant to the drugs listed in
        drugResist.

        drugResist: Which drug resistances to include in the population (a list
        of strings - e.g. ['guttagonol'] or ['guttagonol', 'srinol'])

        returns: The population of viruses (an integer) with resistances to all
        drugs in the drugResist list.
        """

        # TODO
        countResistPop = 0
        for virus in self.viruses:
            isResistant = True
            for drug in drugResist:
                if not virus.isResistantTo(drug):
                    isResistant = False
                    break
            if isResistant:
                countResistPop += 1
        return countResistPop

    def getTotalPop(self):
        return len(self.viruses)

    def update(self):
        """
        Update the state of the virus population in this patient for a single
        time step. update() should execute these actions in order:

        - Determine whether each virus particle survives and update the list of
          virus particles accordingly

        - The current population density is calculated. This population density
          value is used until the next call to update().

        - Based on this value of population density, determine whether each
          virus particle should reproduce and add offspring virus particles to
          the list of viruses in this patient.
          The list of drugs being administered should be accounted for in the
          determination of whether each virus particle reproduces.

        returns: The total virus population at the end of the update (an
        integer)
        """

        # TODO
        for curVirus in self.viruses:
            if random.random() < curVirus.clearProb:
                self.viruses.remove(curVirus) ## self.viruses.remove(curVirus)

        popDensity = 1.0 * len(self.viruses) / self.getMaxPop()

        for i in range( len(self.getViruses()) ):
            curVirus = self.viruses[i]
            # childVirus =  curVirus.reproduce(popDensity)
            # if childVirus != None:
            #     (self.viruses).append(childVirus)
            try:
                self.viruses.append(curVirus.reproduce(popDensity, self.activeDrugs))
            except NoChildException:
                pass

        return len(self.viruses)

## test problem 4
def TestResistantVirus():
    RVirus1 = ResistantVirus(1, 0, {'guttagonol':False, 'srinol':True}, 0.1)
    print "resistances = {}".format( RVirus1.getResistances() )
    print "mutProb = {}".format( RVirus1.getMutProb() )
    print "isResistantTo(\"guttagonol\") = {}".format( RVirus1.isResistantTo('guttagonol') )
    print "isResistantTo(\"srinol\") = {}".format( RVirus1.isResistantTo('srinol') )
    RVirus2 = RVirus1.reproduce( 0, []) ## guttagonol
    print RVirus2 == None

    virus = ResistantVirus(1.0, 0.0, {'drug1':True, 'drug2': True, 'drug3': True, 'drug4': True, 'drug5': True, 'drug6': True}, 0.5)
    for i in range(10):
        childvirus = virus.reproduce(0, [])
        count = 0
        for val in virus.resistances.values():
            if val:
                count += 1
        print "resistant drugs after {} reproduce/mutation: {}".format(i, count/6.)
# TestResistantVirus()
def TestTreatedPatient():
    virus = ResistantVirus(1.0, 0.0, {'drug1':True, 'drug2': True, 'drug3': False, 'drug4': True, 'drug5': True, 'drug6': True}, 0.5)
    Viruses = [virus for i in range(2)]

    virus = ResistantVirus(1.0, 0.0, {}, 0.0)

    P = TreatedPatient([virus], 100)
    # print "Patients add prescription {}".format( P.addPrescription("guttagonol") )
    print "get prescription {}".format( P.getPrescriptions() )
    print "get resistant population {}".format( P.getResistPop(["drug1", "drug3"]) )

    # P.addPrescription("guttagonol")
    for i in range(10):
        P.update()
        print "get viruses population after {} reproduce: {}".format( i, len(P.viruses) )


# TestTreatedPatient()

#
# PROBLEM 5
#
def simulationWithDrug(numViruses, maxPop, maxBirthProb, clearProb, resistances,
                       mutProb, numTrials, timesteps):
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

    """

    # TODO
    avgNumResistPop = []
    avgNumTotalPop = []
    timesteps = 300
    for i in range(numTrials):
        viruses = [ ResistantVirus(maxBirthProb , clearProb, resistances.copy(), mutProb) for ii in range(numViruses)]
        patient = TreatedPatient(viruses, maxPop)
        for j in range(timesteps):
            if i == 0 :
                avgNumResistPop.append(0.)
                avgNumTotalPop.append(0.)
            if j == timesteps - 150:
                patient.addPrescription("guttagonol")
            avgNumTotalPop[j] += patient.update()
            avgNumResistPop[j] += patient.getResistPop(["guttagonol"])
    for i in range(timesteps):
        avgNumTotalPop[i] /= numTrials
        avgNumResistPop[i] /= numTrials

    # print avgNumResistPop
    pylab.plot( avgNumTotalPop, label="Total Virus Population")
    pylab.plot( avgNumResistPop, label="Resistant Virus Population")

    pylab.xlabel("time")
    pylab.ylabel("number of resistant virus")
    pylab.title("SimpleVirus simulation")
    pylab.legend()
    pylab.show()

## test problem 5
# simulationWithDrug(numViruses, maxPop, maxBirthProb, clearProb, resistances, mutProb, numTrials)
#simulationWithDrug(1, 1000, 0.5, 0.2, {"guttagonol": True}, 0.1, 10)
# simulationWithDrug(1, 10, 1.0, 0.0, {}, 1.0, 5)
# simulationWithDrug(1, 100, 0.4, 0.1, {"guttagonol": True}, 0.5, 10)

# Problem set 5
# Problem 1
# Create four histograms (one for each condition of 300, 150, 75, and 0 time step delays). Then, answer the following questions:

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
    fig.text(0.5, 1.0, "Treated Patient simulation", ha='center')
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
simulationDelayedTreatment(10)