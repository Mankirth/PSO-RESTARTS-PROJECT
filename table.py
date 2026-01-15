import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#plt.ioff()

# Box Plot
data = []
line = []

columns = ('PSO Mean', 'PSO StdDev', 'EPSO Mean', 'EPSO StdDev', "%-Diff", "Paired T-Test")
rows = ['1','2','3','4','5','6','7','8','9','10','11','12']
errors = [300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700]
for i in range(12):
    fileResultsDetailsData = pd.read_csv("PSOs 25\\experiment_details.csv")
    objective_name = "F" + str(i+1)
    optimizer_name = "PSO"
    detailedData = fileResultsDetailsData[
        (fileResultsDetailsData["Optimizer"] == optimizer_name)
        & (fileResultsDetailsData["objfname"] == objective_name)
    ]
    detailedData = detailedData["Iter" + str(2000)]
    detailedData = np.array(detailedData).T.tolist()
    data.append(detailedData)
    for t in range(30):
        detailedData[t] -= errors[i]
    gwoData = detailedData.copy()

    fileResultsDetailsData = pd.read_csv("PSOs 25\\experiment_details.csv")
    objective_name = "F" + str(i+1)
    optimizer_name = "EPSO"
    detailedData = fileResultsDetailsData[
        (fileResultsDetailsData["Optimizer"] == optimizer_name)
        & (fileResultsDetailsData["objfname"] == objective_name)
    ]
    detailedData = detailedData["Iter" + str(2000)]
    detailedData = np.array(detailedData).T.tolist()
    data.append(detailedData)

    for t in range(30):
        detailedData[t] -= errors[i]
    gwomData = detailedData.copy()

    total = 0
    sampleDiffs = 0
    diffs = np.zeros(30)
    for t in range(30):
        sampleDiffs += gwoData[t] - gwomData[t]
    sampleMean = sampleDiffs / 30
    #sampleMean = np.mean(gwomData) - np.mean(gwoData)
    for t in range(30):
        total += np.pow(((gwoData[t]-gwomData[t]) - sampleMean), 2)
    sampleStd = np.sqrt(total / (29))
    testStat = sampleMean/(sampleStd/np.sqrt(30))
    percentDiff = ((1 - (np.mean(gwomData)/np.mean(gwoData))) * 100).round(4)
    if(i+1 == 1):
        percentDiff = -1*((1 - (np.mean(gwoData)/np.mean(gwomData))) * 100).round(4)
    line.append([np.mean(gwoData).round(4), np.std(gwoData).round(4), np.mean(gwomData).round(4), np.std(gwomData).round(4), percentDiff, testStat.round(4)])
cellText = line

# , notch=True

fig, ax = plt.subplots()

# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(rowLabels=rows,
        colLabels=columns,
        cellText=cellText,
        loc='center')
fig.tight_layout()

fig_name = "PSOs 25\\" + "table.png"
plt.savefig(fig_name, bbox_inches="tight")
plt.clf()