import sys
import random
def generateRandomNode(dimension, min, max):
    point = []
    for i in range(0,dimension):
        point.append(random.randint(min,max))
    return point

def exists(node, nodes_list):
    if node in nodes_list:
        return False
    return True    

def main():
    if len(sys.argv) != 5:
        print("Incorrect number of parameters: \n"+
              "\t generateInstance.py \"dimension\" \"N nodes\" \"Min_value\" \"Max_value\"")
        return -1

    dim = int(sys.argv[1])
    n_nodes = int(sys.argv[2])
    min_val = int(sys.argv[3])
    max_val = int(sys.argv[4])
    
    file_name = 'instance_'+str(dim)+'_'+str(n_nodes)+'_'+str(min_val)+'_'+str(max_val)+'.csv'
    
    
    f = open(file_name, "w+")
    nodes_list = []
    for i in range(n_nodes):
        random_node = generateRandomNode(dim, min_val, max_val)
        f.write(str(random_node[0])+','+str(random_node[1])+'\n')
    f.close()
    
if __name__=="__main__":
    main()