class Frob(object):
    def __init__(self, name):
        self.name = name
        self.before = None
        self.after = None
    def setBefore(self, before):
        # example: a.setBefore(b) sets b before a
        self.before = before
    def setAfter(self, after):
        # example: a.setAfter(b) sets b after a
        self.after = after
    def getBefore(self):
        return self.before
    def getAfter(self):
        return self.after
    def myName(self):
        return self.name
def insert(atMe, newFrob):
    """
    atMe: a Frob that is part of a doubly linked list
    newFrob:  a Frob with no links 
    This procedure appropriately inserts newFrob into the linked list that atMe is a part of.
    """
    newName = newFrob.myName()
    curFrob = atMe
    def insertPos(curFrob, newName, toLeft):
        curName = curFrob.myName()
        insertFrob = curFrob
        insertAfter = True
        if toLeft:
            if newName >= curName:
                return [insertFrob, insertAfter]
            else:
                if curFrob.getBefore() == None:
                    return [insertFrob, not insertAfter]
                else:
                    return insertPos(curFrob.getBefore(), newName, toLeft)
        else:
            if newName < curName:
                #print "{}_{}".format(newName,curName)
                return [insertFrob, not insertAfter]
            else:
                if curFrob.getAfter() == None:
                    #print "{}_None".format(curName)
                    return [insertFrob, insertAfter]
                else:
                    return insertPos(curFrob.getAfter(), newName, toLeft)
    if newName <= curFrob.myName():
        insertFrob, insertAfter = insertPos(curFrob, newName, True)
    else:
        insertFrob, insertAfter = insertPos(curFrob, newName, False)

    if insertAfter:
        afterFrob = insertFrob.getAfter()
        if afterFrob != None :
            afterFrob.setBefore(newFrob)
            newFrob.setAfter(afterFrob)
        insertFrob.setAfter(newFrob)
        newFrob.setBefore(insertFrob)
    else:
        beforeFrob = insertFrob.getBefore()
        if beforeFrob != None:
            beforeFrob.setAfter(newFrob)
            newFrob.setBefore(beforeFrob)
        insertFrob.setBefore(newFrob)
        newFrob.setAfter(insertFrob)
def findFront(start):
    """
    start: a Frob that is part of a doubly linked list
    returns: the Frob at the beginning of the linked list
    """
    # Your Code Here
    if start.getBefore() == None:
        return start
    else:
        return findFront(start.getBefore())

eric = Frob('eric')
andrew = Frob('andrew')
ruth = Frob('ruth')
fred = Frob('fred')
martha = Frob('martha')
insert(eric, eric)
insert(eric, andrew)
insert(eric, ruth)
insert(eric, fred)
insert(ruth, martha)
insert(eric, Frob("mark"))
insert(eric, Frob('martha'))

def walk(atMe):
    headFrob = atMe
    while True:
        if headFrob.getBefore() != None:
            headFrob = headFrob.getBefore()
        else:
            break
    while True:
        print headFrob.myName()
        if headFrob.getAfter() != None:
            headFrob = headFrob.getAfter()
        else:
            break
walk(andrew)