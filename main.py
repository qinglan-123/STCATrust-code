import time

import numpy as np
# Import GCNTrainer class from gcn module, used to train and evaluate STCATrust model
from gcn import GCNTrainer
# Import parameter_parser function from arg_parser module, used to parse command-line arguments
from arg_parser import parameter_parser

from utils import tab_printer, read_graph, best_printer

def main():
    t0 = time.time()

    # Call parameter_parser function to parse command-line arguments, and store the result in args variable
    args = parameter_parser()
    # Call tab_printer function to print parsed command-line parameters in table format
    tab_printer(args)

    edges = read_graph(args)  # number of edges --> otc: 35592, alpha: 24186

    best = [["# Training Timeslots", "Epoch", "MCC", "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro", "Run Time"]]

    times = 8 if args.single_prediction else 6

    for t in range(times):
        # Create an instance of GCNTrainer class, passing in command-line parameters args and graph data edges
        trainer = GCNTrainer(args, edges)
        # Call trainer's setup_dataset method to set up training and test datasets
        trainer.setup_dataset()

        # Print prompt information indicating the current round of runs
        print("Ready, Go! Round = " + str(t))
        # Call trainer's create_and_train_model method to start training and evaluating the model
        trainer.create_and_train_model()

        # Initialize a list best_epoch, used to store the best evaluation results of the current run
        best_epoch = [0, 0, 0, 0, 0, 0, 0]
        # Iterate through each element in trainer.logs["performance"] list (starting from the second element)
        for i in trainer.logs["performance"][1:]:

            if float(i[1]+i[2]+i[3]+i[6]) > (best_epoch[1]+best_epoch[2]+best_epoch[3]+best_epoch[6]):
                best_epoch = i

        # Add the total training time of the current run to the end of the best_epoch list
        best_epoch.append(trainer.logs["training_time"][-1][1])
        # Insert the current training time slot number (t + 2) at the beginning of the best_epoch list
        best_epoch.insert(0, t + 2)
        # Add the updated best_epoch list to the best list
        best.append(best_epoch)

        # Increase the training time slot count in command-line parameters, preparing for the next run
        args.train_time_slots += 1

    # Print prompt information indicating that the best results of each run will be printed next
    print("\nBest results of each run")
    # Call best_printer function to print the best results in the best list in table format
    best_printer(best)

    # Print prompt information indicating that the mean, max, min and standard deviation of evaluation metrics will be printed next
    print("\nMean, Max, Min, Std")
    # Convert the best list to numpy array, remove the header, then convert the data type to float64
    analyze = np.array(best)[1:, 1:].astype(np.float64)
    # Calculate the mean of evaluation metrics
    mean = np.mean(analyze, axis=0)
    # Calculate the max of evaluation metrics
    maxi = np.amax(analyze, axis=0)
    # Calculate the min of evaluation metrics
    mini = np.amin(analyze, axis=0)
    # Calculate the standard deviation of evaluation metrics
    std = np.std(analyze, axis=0)
    # Create a new list results, containing the header and calculated mean, max, min and standard deviation
    results = [["Epoch", 'MCC', "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro", "Run Time"], mean, maxi, mini, std]

    # Call best_printer function to print statistical results in the results list in table format
    best_printer(results)

    print("Running time:",time.time()-t0)

    # Save best results as CSV file
    best_df = pd.DataFrame(best[1:], columns=best[0])
    best_df.to_csv('best_results.csv', index=False)

    # Save statistical results as CSV file
    results_df = pd.DataFrame(results[1:], columns=results[0])
    results_df.to_csv('statistics_results.csv', index=False)

if __name__ == "__main__":
    main()
    