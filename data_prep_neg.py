import numpy as np
from qiskit import *
import csv

# Declare input range
a = np.arange(0.0, 1.01, 0.01)

# Declare t-norms to test
negs = np.array([
  lambda a: 1 - a,                                                        # Standard
  lambda a: 1 - (1 - 0.1)/0.1 * a if a <= 0.1 else 0.1/(1 - 0.1)*(1 - a), # Knee-threshold(0.1)
  lambda a: 1 - (1 - 0.3)/0.3 * a if a <= 0.3 else 0.3/(1 - 0.3)*(1 - a), # Knee-threshold(0.3)
  lambda a: 1 - (1 - 0.5)/0.5 * a if a <= 0.5 else 0.5/(1 - 0.5)*(1 - a), # Knee-threshold(0.5)
  lambda a: 1 - (1 - 0.7)/0.7 * a if a <= 0.7 else 0.7/(1 - 0.7)*(1 - a), # Knee-threshold(0.7)
  lambda a: 1 - (1 - 0.9)/0.9 * a if a <= 0.9 else 0.9/(1 - 0.9)*(1 - a), # Knee-threshold(0.9)
  lambda a: (1 - a)/(1 + 10*a),                                           # Sugeno(10)
  lambda a: (1 - a)/(1 + 2*a),                                            # Sugeno(2)
  lambda a: (1 - a)/(1 + 0*a),                                            # Sugeno(0)
  lambda a: (1 - a)/(1 + -0.5*a),                                         # Sugeno(0.5)
  lambda a: (1 - a)/(1 + -0.9*a),                                         # Sugeno(0.9)
  lambda a: np.power(1 - np.power(a,0.2), 1/0.2),                         # Yager (0.2)
  lambda a: np.power(1 - np.power(a,0.5), 1/0.5),                         # Yager (0.5)
  lambda a: np.power(1 - np.power(a,1), 1/1),                             # Yager (1)
  lambda a: np.power(1 - np.power(a,2), 1/2),                             # Yager (2)
  lambda a: np.power(1 - np.power(a,5), 1/5),                             # Yager (5)
  lambda a: 1/2 * (1 + np.cos(np.pi * a)),                                # Cosine
])

# Declare outputs
z = np.zeros((negs.size + 1, a.size))

# Calculate outputs
for ix, x in enumerate(a):
    # For each t-norm
    for it, neg in enumerate(negs):
      z[it][ix] = neg(x)
    
    # Quantum t-norm
    qbits = QuantumRegister(1)
    out = ClassicalRegister(1)
    circ = QuantumCircuit(qbits, out)

    circ.ry(np.pi*x, qbits[0])
    circ.x(qbits[0])
    circ.measure(qbits[0], out[0])

    aer_sim = Aer.get_backend('aer_simulator')
    job = aer_sim.run(circ, shots=10000)
    hist = job.result().get_counts()
    try:
      z[len(negs)][ix] = hist['1']/10000
    except KeyError:
      z[len(negs)][ix] = 0

# Prepare data
tags = [
  'a',
  'standard',
  'knee-threshold_0.1',
  'knee-threshold_0.3',
  'knee-threshold_0.5',
  'knee-threshold_0.7',
  'knee-threshold_0.9',
  'sugeno_10',
  'sugeno_2',
  'sugeno_0',
  'sugeno_0.5',
  'sugeno_0.9',
  'yager_0.2',
  'yager_0.5',
  'yager_1',
  'yager_2',
  'yager_5',
  'cosine',
  'quantum'
]

with open('data/data_neg.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONE)

    # write the header
    writer.writerow(tags)

    # write the data
    for ix, x in enumerate(a):
        data = []
        data.append(x)
        for it, tnorm in enumerate(negs):
          data.append(z[it][ix])
        data.append(z[len(negs)][ix])
        writer.writerow(['{:.5f}'.format(d) for d in data])