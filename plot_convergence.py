import matplotlib.pyplot as plt
import pandas as pd


#def run(results_directory, optimizer, objectivefunc, Iterations):
results_directory = "Finals CEC"
objectivefunc = ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12"]
plt.ioff()
fileResultsData = pd.read_csv(results_directory + "/experiment.csv")

for j in range(0, len(objectivefunc)):
    objective_name = objectivefunc[j]

    startIteration = 0
    Iterations = 2000
    allGenerations = [x + 1 for x in range(startIteration, Iterations)]
    optimizer_name = "FPSO"

    row = fileResultsData[
        (fileResultsData["Optimizer"] == optimizer_name)
        & (fileResultsData["objfname"] == objective_name)
    ]
    row = row.iloc[:, 3 + startIteration :]

    plt.plot(allGenerations, row.values.tolist()[0], label=optimizer_name)

    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.02))
    plt.grid()
    fig_name = results_directory + "\\Separated\\5PSO/convergence-" + objective_name + ".png"
    plt.savefig(fig_name, bbox_inches="tight")
    plt.clf()
    # plt.show()
