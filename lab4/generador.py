import numpy as np
N = 3    # Numero de columnas
M = 3    # Numero de filas

array = np.random.randint(10, size= N*M)
print (array)

a = open('initial.txt', 'w')
a.write(str(N)+' '+str(M)+'\n')
a.write(' '.join(str(e) for e in array))
a.close()