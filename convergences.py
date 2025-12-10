import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

data = np.zeros((30, 6000))
fileResultsDetailsData = pd.read_csv("EPSO collected\\experiment_details.csv")
objective_name = "rastrigin"
optimizer_name = "EPSO"

detailedData = fileResultsDetailsData[
    (fileResultsDetailsData["Optimizer"] == optimizer_name)
    & (fileResultsDetailsData["objfname"] == objective_name)
]
for i in range(1, 6001):
    columnData = detailedData["Iter" + str(i)]
    columnData = np.array(columnData).T.tolist()
    for t in range(0, 30):
        data[t][i-1] = columnData[t]

for t in range(0, 30):
    plt.plot(data[t].tolist())
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.02))
    plt.grid()
    fig_name = "Convergences\\EPSO Collected\\/EPSO-Convergence"+ str(t) + "-" + objective_name + ".png"
    plt.savefig(fig_name, bbox_inches="tight")
    plt.clf()

# plt.gca().set_ylim([0, 250])
# fig_name = "Tests 30x50x6000 V2\\GWO\\" + "/GWO Both Shift Comparison" + ".png"
# plt.savefig(fig_name, bbox_inches="tight")
# plt.clf()