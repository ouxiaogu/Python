import random
import pylab

# Global Variables
MAXRABBITPOP = 1000
CURRENTRABBITPOP = 500
CURRENTFOXPOP = 30

def rabbitGrowth():
    """
    rabbitGrowth is called once at the beginning of each time step.

    It makes use of the global variables: CURRENTRABBITPOP and MAXRABBITPOP.

    The global variable CURRENTRABBITPOP is modified by this procedure.

    For each rabbit, based on the probabilities in the problem set write-up,
      a new rabbit may be born.
    Nothing is returned.
    """
    # you need this line for modifying global variables
    global CURRENTRABBITPOP
    reproduceRate = 1 - 1.*CURRENTRABBITPOP/MAXRABBITPOP
    for i in range(CURRENTRABBITPOP):
        if random.random() <= reproduceRate :
            if CURRENTRABBITPOP >= 10 and CURRENTRABBITPOP < MAXRABBITPOP:
                CURRENTRABBITPOP += 1
    # TO DO
    pass

def foxGrowth():
    """
    foxGrowth is called once at the end of each time step.

    It makes use of the global variables: CURRENTFOXPOP and CURRENTRABBITPOP,
        and both may be modified by this procedure.

    Each fox, based on the probabilities in the problem statement, may eat
      one rabbit (but only if there are more than 10 rabbits).

    If it eats a rabbit, then with a 1/3 prob it gives birth to a new fox.

    If it does not eat a rabbit, then with a 1/10 prob it dies.

    Nothing is returned.
    """
    # you need these lines for modifying global variables
    global CURRENTRABBITPOP
    global CURRENTFOXPOP
    ateRabitRate = 1.*CURRENTRABBITPOP/MAXRABBITPOP
    numFox = CURRENTFOXPOP
    for i in range(numFox):
        if random.random() <= ateRabitRate :
            if CURRENTRABBITPOP > 10:
                CURRENTRABBITPOP -= 1
                if random.random() <= 1./3:
                    CURRENTFOXPOP += 1
        elif random.random() < 9./10:
            if CURRENTFOXPOP > 10 :
                CURRENTFOXPOP -= 1
    # TO DO
    pass


def runSimulation(numSteps):
    """
    Runs the simulation for `numSteps` time steps.

    Returns a tuple of two lists: (rabbit_populations, fox_populations)
      where rabbit_populations is a record of the rabbit population at the
      END of each time step, and fox_populations is a record of the fox population
      at the END of each time step.

    Both lists should be `numSteps` items long.
    """
    assert( CURRENTFOXPOP >= 10 and CURRENTRABBITPOP >=10)
    rabbit_populations = []
    fox_populations = []
    for  i in range(numSteps):
        rabbitGrowth()
        foxGrowth()
        rabbit_populations.append(CURRENTRABBITPOP)
        fox_populations.append(CURRENTFOXPOP)
    return (rabbit_populations, fox_populations)
    # TO DO
    pass

def plotSim():
    numSim = 400
    rabbit_populations, fox_populations = runSimulation(numSim)
    pylab.figure(1)
    pylab.plot(range(numSim), rabbit_populations, label = "rabit")
    pylab.plot(range(numSim), fox_populations, label = "fox")

#    pylab.figure(2)
    coeff = pylab.polyfit(range(numSim), rabbit_populations, 2)
    pylab.plot(pylab.polyval(coeff, range(numSim)), label = "rabit fitting")
    coeff = pylab.polyfit(range(numSim), fox_populations, 2)
    pylab.plot(pylab.polyval(coeff, range(numSim)), label = "fox fitting") 
    
    pylab.legend()
    pylab.show()

plotSim()


#for n in range(10):
#    CURRENTRABBITPOP = 1000
#    MAXRABBITPOP = 1000
#    CURRENTFOXPOP = 50 
#    foxGrowth()
#    print CURRENTRABBITPOP, CURRENTFOXPOP
#for n in range(5):    
#    MAXRABBITPOP = 1000
#    CURRENTRABBITPOP = 500 
#    CURRENTFOXPOP = 0 
#    rabbitGrowth()
#    print CURRENTRABBITPOP, CURRENTFOXPOP