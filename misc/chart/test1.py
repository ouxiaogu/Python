d = {}

keyList1 = ["Person", "Male", "Boy", "Student", "id", "Name", "Name"]
ValList1 = ["1", "Y", "Y", "Roger", "", "Roger", "Roger2"]

length = len(keyList1)

def insert(cur, key, value):
    if not cur.has_key(key):
        cur[key] = value
        
for i in range(length):
    insert(d, keyList1[i], ValList1[i])
e = d
print(e)

a =1024
print("{} and {}".format("string", a))


    