import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics

data = pd.read_csv('data/data_neg.csv')
tags = data.columns.values.tolist()[1:]

plt.xlim((0,1))
plt.ylim((0,1))

plt.grid(visible=True, which='major', zorder=-1.0)

plt.gca().set_aspect('equal')

clear = False
last_clear = ''
for it, tag in enumerate(tags):
  clear = ('standard' in tag or 'cosine' in tag or 'quantum' in tag)
  if clear:
    plt.clf()
  else:
    if last_clear == '':
      last_clear = tag.split('_')[0]
    elif not last_clear in tag:
      plt.clf()
      last_clear = tag.split('_')[0]
  plt.plot(data.a, data[tag], color='C0')
  if not clear:
    pos = statistics.mean(data[abs(data.a - data[tag]) < 0.05][tag]) + .01
    plt.annotate('β={}'.format(tag.split('_')[1]), xy=(pos,pos))
  plt.grid(visible=True)
  plt.gca().set_aspect('equal')
  plt.savefig('figs/neg/{}.pdf'.format(tag), bbox_inches='tight')

