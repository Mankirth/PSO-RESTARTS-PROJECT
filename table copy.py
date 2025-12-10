import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#plt.ioff()

# Box Plot
data = []
line = []

fileResultsDetailsData = pd.read_csv("PSO 2.05 with clamp\\experiment_details.csv")
objective_name = "rastrigin"
optimizer_name = "PSO"
detailedData = fileResultsDetailsData[
    (fileResultsDetailsData["Optimizer"] == optimizer_name)
    & (fileResultsDetailsData["objfname"] == objective_name)
]
detailedData = detailedData["Iter" + str(5999)]
detailedData = np.array(detailedData).T.tolist()
data.append(detailedData)

gwomData = detailedData.copy()

print(np.mean(gwomData).round(4))
print(np.std(gwomData).round(4))

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

fig_name = "PSOs F1-F12\\" + "table.png"
plt.savefig(fig_name, bbox_inches="tight")
plt.clf()