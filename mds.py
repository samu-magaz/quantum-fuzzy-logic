import sys
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

NORM = sys.argv[1]

tags = {
    'tnorm': ['Minimum','Product','Łukasiewicz','Nilpotent\nminimum','Hamacher\nproduct','Quantum'],
    'tconorm': ['Maximum','Probabilistic\nsum','Bounded\nsum','Nilpotent\nmaximum','Einstein\nsum','Quantum'],
}

df = pd.read_csv('data/data_{}.csv'.format(NORM))
df = df.T
df = df[2:]

#perform multi-dimensional scaling
mds = MDS(random_state=0)
scaled_df = mds.fit_transform(df)

#create scatterplot
plt.scatter(scaled_df[:,0], scaled_df[:,1])

#add axis labels
# plt.xlabel('Coordinate 1')
# plt.ylabel('Coordinate 2')

plt.xlim((-2.6,2.6))
plt.ylim((-2.6,2.6))

plt.grid(visible=True, which='major', zorder=-1.0)

plt.gca().set_aspect('equal')

#add lables to each point
for i, txt in enumerate(tags[NORM]):
    plt.annotate(txt, (scaled_df[:,0][i], scaled_df[:,1][i]-.3), verticalalignment='center', horizontalalignment='center')

#display scatterplot
plt.savefig('figs/mds/mds_{}.pdf'.format(NORM), bbox_inches='tight')
