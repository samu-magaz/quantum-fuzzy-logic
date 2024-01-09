import numpy as np
from qiskit import *
import csv

# Declare input range
a = np.arange(0.0, 1.01, 0.05)
b = np.arange(0.0, 1.01, 0.05)

# Declare t-norms to test
tnorms = np.array([
  lambda a,b: min(a, b),                                   # Minimun
  lambda a,b: a * b,                                       # Product
  lambda a,b: max(0, a + b - 1),                           # Lukasiewicz
  lambda a,b: min(a, b) if (a + b) > 1 else 0,             # Nilpotent minimum
  lambda a,b: 0 if a == b == 0 else (a * b)/(a + b - a*b), # Hamacher product
])

# Declare outputs
z = np.zeros((tnorms.size + 1, a.size, b.size))

# Calculate outputs
for ix, x in enumerate(a):
  for iy, y in enumerate(b):
    # For each t-norm
    for it, tnorm in enumerate(tnorms):
      z[it][ix][iy] = tnorm(x, y)
    
    # Quantum t-norm
    qbits = QuantumRegister(3)
    out = ClassicalRegister(1)
    circ = QuantumCircuit(qbits, out)

    circ.ry(np.pi*x, qbits[0])
    circ.ry(np.pi*y, qbits[1])
    circ.ccx(qbits[0], qbits[1], qbits[2])
    circ.measure(qbits[2], out[0])

    aer_sim = Aer.get_backend('aer_simulator')
    job = aer_sim.run(circ, shots=10000)
    hist = job.result().get_counts()
    try:
      z[len(tnorms)][ix][iy] = hist['1']/10000
    except KeyError:
      z[len(tnorms)][ix][iy] = 0

# Prepare data
tags = [
  'a',
  'b',
  'minimum',
  'product',
  'lukasiewicz',
  'nilpotent_minimum',
  'hamacher_product',
  'quantum'
]

with open('data/data_tnorm.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONE)

    # write the header
    writer.writerow(tags)

    # write the data
    for ix, x in enumerate(a):
      for iy, y in enumerate(b):
        data = []
        data.append(x)
        data.append(y)
        for it, tnorm in enumerate(tnorms):
          data.append(z[it][ix][iy])
        data.append(z[len(tnorms)][ix][iy])
        writer.writerow(['{:.5f}'.format(d) for d in data])