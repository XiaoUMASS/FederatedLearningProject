import numpy as np
import os.path
import matplotlib

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
from pathlib import Path

METRIC = "f1"  # "accuracy", "f1"
METRIC_COLUMNS = {
    "accuracy": 5,
    "f1": 6,
}
METRIC_INDEX = METRIC_COLUMNS[METRIC]
DATASETS = [
    "Opp",
    "mHealth",
    "UR_Fall",
]

ROUND_CUT = {
    "Opp": {"single": 50,
            "cross": 50},
    "mHealth": {"single": 10,
                "cross": 25},
    "UR_Fall": {"single": 100,
                "cross": 100},
}

aes = {
    "Opp": ["dccae"],
    "mHealth": ["split_ae"],
    "UR_Fall": ["split_ae"],
}
ae_print = {
    "split_ae": "SplitAE",
    "dccae": "DCCAE",
}
combo = {
    "Opp": [("acce", "gyro")],
    "mHealth": [("acce", "gyro"), ("acce", "mage"), ("gyro", "mage")],
    "UR_Fall": [("acce", "depth"), ("rgb", "depth")],
}

cross_selected = {
    "Opp": {"acce_gyro": [
        "client_A_label_A_test_A",
        "client_AB_label_B_test_A",
        "client_ABA_label_B_test_A",
        "ablation_label_B_test_A",
        "client_B_label_B_test_B",
        "client_AB_label_A_test_B",
        "client_ABAB_label_A_test_B",
        "ablation_label_A_test_B",
    ]
    },
    "mHealth": {"acce_gyro": [
        "client_A_label_A_test_A",
        "client_AB_label_B_test_A",
        "client_ABB_label_B_test_A",
        "ablation_label_B_test_A",
        "client_B_label_B_test_B",
        "client_AB_label_A_test_B",
        "client_ABB_label_A_test_B",
        "ablation_label_A_test_B",
    ],
        "acce_mage": [
        "client_A_label_A_test_A",
        "client_AB_label_B_test_A",
        "client_ABB_label_B_test_A",
        "ablation_label_B_test_A",
        "client_B_label_B_test_B",
        "client_AB_label_A_test_B",
        "client_ABA_label_A_test_B",
        "ablation_label_A_test_B",
    ],
        "gyro_mage": [
        "client_A_label_A_test_A",
        "client_AB_label_B_test_A",
        "client_ABA_label_B_test_A",
        "ablation_label_B_test_A",
        "client_B_label_B_test_B",
        "client_AB_label_A_test_B",
        "client_ABB_label_A_test_B",
        "ablation_label_A_test_B",
    ]},
    "UR_Fall": {"acce_depth": [
        "client_A_label_A_test_A",
        "client_AB_label_B_test_A",
        "client_ABA_label_B_test_A",
        "ablation_label_B_test_A",
        "client_B_label_B_test_B",
        "client_AB_label_A_test_B",
        "client_ABB_label_A_test_B",
        "ablation_label_A_test_B",
    ],
        "rgb_depth": [
        "client_A_label_A_test_A",
        "client_AB_label_B_test_A",
        "client_ABAB_label_B_test_A",
        "ablation_label_B_test_A",
        "client_B_label_B_test_B",
        "client_AB_label_A_test_B",
        "client_ABAB_label_A_test_B",
        "ablation_label_A_test_B",

    ]}
}

modality_print = {
    "acce": "Acce",
    "gyro": "Gyro",
    "mage": "Mag",
    "rgb": "RGB",
    "depth": "Depth",
}
N_REPS = 1

CB_color_cycle = ['#ff7f00', '#377eb8',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00', '#4daf4a']


def single_multi_modality_comparison():
    for dataset in DATASETS:
        for ae in aes[dataset]:
            for modalities in combo[dataset]:
                plt.figure()
                fig, ax = plt.subplots(
                    nrows=1, ncols=1, figsize=(5, 3.5))

                ax.set_ylim([0.0, 0.8])
                plt.xticks(fontsize="large")
                plt.yticks(fontsize="large")

                schemes = {"client_A_label_A_test_A": "A30_B0_AB0_label_A_test_A",
                           "client_B_label_B_test_B": "A0_B30_AB0_label_B_test_B",
                           "client_AB_label_AB_test_A": "A0_B0_AB30_label_AB_test_A",
                           "client_AB_label_AB_test_B": "A0_B0_AB30_label_AB_test_B", }
                legends = {"client_A_label_A_test_A": "$\mathregular{UmFL_A}$",
                           "client_B_label_B_test_B": "$\mathregular{UmFL_B}$",
                           "client_AB_label_AB_test_A": "$\mathregular{MmFL_{AB}-L_{AB}-T_A}$",
                           "client_AB_label_AB_test_B": "$\mathregular{MmFL_{AB}-L_{AB}-T_B}$", }
                for k in schemes:
                    color_idx = 0 if k[-1] == "A" else 1
                    linestyle = ("-" if color_idx == 0 else "dashdot")if k[8] != "_" else (
                        "--" if color_idx == 0 else (0, (5, 10)))
                    test_acc = []
                    for rep in range(N_REPS):
                        rep_file = os.path.join("results", dataset.lower(
                        ), ae, f"{modalities[0]}_{modalities[1]}", schemes[k], "results.txt")  #), ae, f"{modalities[0]}_{modalities[1]}", schemes[k], f"rep_{rep}", "results.txt")
                        data = np.loadtxt(rep_file, delimiter=",")
                        x_all = data[:, 0]
                        idxs_round_cut = x_all <= ROUND_CUT[dataset]["single"]
                        x = x_all[idxs_round_cut]
                        y_rep = data[idxs_round_cut, METRIC_INDEX]
                        y_rep[y_rep == 0.0] = np.nan
                        test_acc.append(y_rep)
                    y = np.nanmean(np.array(test_acc), axis=0)
                    ax.plot(
                        x, y, color=CB_color_cycle[color_idx], linestyle=linestyle, label=legends[k])
                    se = stats.sem(np.array(test_acc), nan_policy="omit")
                    ax.fill_between(x, y-se, y+se,
                                    color=CB_color_cycle[color_idx], alpha=0.3)
                ax.set_xlabel("Communication rounds", fontsize="x-large")
                metric_label = "accuracy" if METRIC == "accuracy" else "$\mathregular{F_1}$"
                ax.set_ylabel(
                    f"Test {metric_label}", fontsize="x-large")
                dataset_print = dataset.replace("_", " ")
                ax.set_title(
                    f"{dataset_print}, {ae_print[ae]}, A: {modality_print[modalities[0]]}, B: {modality_print[modalities[1]]}")
                ax.legend(loc="lower right", fontsize="x-large")
                Path("plots").mkdir(parents=True, exist_ok=True)
                plt.savefig(f"plots/single_multi_modality_comparison_{dataset}_{ae}_{modalities[0]}_{modalities[1]}.pdf",
                            bbox_inches="tight")


def cross_modality_comparison():
    for dataset in DATASETS:
        for ae in aes[dataset]:
            for modalities in combo[dataset]:
                plt.figure()
                fig, axes = plt.subplots(
                    nrows=1, ncols=2, figsize=(10, 3.7))

                schemes_test_A = {
                    "client_A_label_A_test_A": (0, "A30_B0_AB0_label_A_test_A"),
                    "client_AB_label_B_test_A": (1, "A0_B0_AB30_label_B_test_A"),
                    "client_ABA_label_B_test_A": (2, "A10_B0_AB30_label_B_test_A"),
                    "client_ABB_label_B_test_A": (3, "A0_B10_AB30_label_B_test_A"),
                    "client_ABAB_label_B_test_A": (4, "A10_B10_AB30_label_B_test_A"),
                    "ablation_label_B_test_A": (5, "A30_B30_AB0_label_B_test_A"),
                }
                schemes_test_B = {
                    "client_B_label_B_test_B": (0, "A0_B30_AB0_label_B_test_B"),
                    "client_AB_label_A_test_B": (1, "A0_B0_AB30_label_A_test_B"),
                    "client_ABB_label_A_test_B": (2, "A0_B10_AB30_label_A_test_B"),
                    "client_ABA_label_A_test_B": (3, "A10_B0_AB30_label_A_test_B"),
                    "client_ABAB_label_A_test_B": (4, "A10_B10_AB30_label_A_test_B"),
                    "ablation_label_A_test_B": (5, "A30_B30_AB0_label_A_test_B"),
                }
                legends = {
                    "client_A_label_A_test_A": "$\mathregular{UmFL_A}$",
                    "client_AB_label_B_test_A": "$\mathregular{MmFL_{AB}-L_B-T_A}$",
                    "client_ABA_label_B_test_A": "$\mathregular{MmFL_{ABA}-L_B-T_A}$",
                    "client_ABB_label_B_test_A": "$\mathregular{MmFL_{ABB}-L_B-T_A}$",
                    "client_ABAB_label_B_test_A": "$\mathregular{MmFL_{ABAB}-L_B-T_A}$",
                    "ablation_label_B_test_A": "$\mathregular{Abl-L_B-T_A}$",
                    "client_B_label_B_test_B": "$\mathregular{UmFL_B}$",
                    "client_AB_label_A_test_B": "$\mathregular{MmFL_{AB}-L_A-T_B}$",
                    "client_ABB_label_A_test_B": "$\mathregular{MmFL_{ABB}-L_A-T_B}$",
                    "client_ABA_label_A_test_B": "$\mathregular{MmFL_{ABA}-L_A-T_B}$",
                    "client_ABAB_label_A_test_B": "$\mathregular{MmFL_{ABAB}-L_A-T_B}$",
                    "ablation_label_A_test_B": "$\mathregular{Abl-L_A-T_B}$",
                }

                groups = (schemes_test_A, schemes_test_B)
                for col, schemes in enumerate(groups):
                    ax = axes[col]
                    ax.set_ylim([0.0, 0.7 if dataset == "UR_Fall" else 0.8])
                    plt.xticks(fontsize="large")
                    plt.yticks(fontsize="large")
                    for k in schemes:
                        if k not in cross_selected[dataset][f"{modalities[0]}_{modalities[1]}"]:
                            continue
                        color_idx = schemes[k][0]
                        linestyle = (
                            "-" if color_idx == 1 else "dashdot") if color_idx != 0 and color_idx != 5 else "--" if color_idx == 0 else "dotted"
                        test_acc = []
                        for rep in range(N_REPS):
                            if "ablation" not in k:
                                rep_file = os.path.join(
                                    "results", dataset.lower(), ae, f"{modalities[0]}_{modalities[1]}", schemes[k][1], "results.txt") #"results", dataset.lower(), ae, f"{modalities[0]}_{modalities[1]}", schemes[k][1], f"rep_{rep}", "results.txt")
                            else:
                                rep_file = os.path.join(
                                    "results", dataset.lower(), "ablation", f"{modalities[0]}_{modalities[1]}", schemes[k][1], "results.txt") #"results", dataset.lower(), "ablation", f"{modalities[0]}_{modalities[1]}", schemes[k][1], f"rep_{rep}", "results.txt")
                            data = np.loadtxt(rep_file, delimiter=",")
                            x_all = data[:, 0]
                            idxs_round_cut = x_all <= ROUND_CUT[dataset]["cross"]
                            x = x_all[idxs_round_cut]
                            y_rep = data[idxs_round_cut, METRIC_INDEX]
                            y_rep[y_rep == 0.0] = np.nan
                            test_acc.append(y_rep)
                        y = np.nanmean(np.array(test_acc), axis=0)
                        ax.plot(
                            x, y, color=CB_color_cycle[color_idx], linestyle=linestyle, label=legends[k])
                        se = stats.sem(np.array(test_acc), nan_policy="omit")
                        ax.fill_between(x, y-se, y+se,
                                        color=CB_color_cycle[color_idx], alpha=0.3)
                    ax.set_xlabel("Communication rounds", fontsize="x-large")
                    metric_label = "accuracy" if METRIC == "accuracy" else "$\mathregular{F_1}$"
                    ax.set_ylabel(
                        f"Test {metric_label}", fontsize="x-large")
                    dataset_print = dataset.replace("_", " ")
                    ax.set_title(
                        f"{dataset_print}, {ae_print[ae]}, A: {modality_print[modalities[0]]}, B: {modality_print[modalities[1]]}")
                    ax.legend(loc="lower right", fontsize="x-large")
                Path("plots").mkdir(parents=True, exist_ok=True)
                plt.savefig(f"plots/cross_modality_comparison_{dataset}_{ae}_{modalities[0]}_{modalities[1]}.pdf",
                            bbox_inches="tight")

def Train_Accuracy_Per_Class():
    title =  "Per-Class Train Accuracy"
    xlabel = "Round"
    ylabel = "Accuracy"

    # Get paths to each parent folder containing result data
    result_modalities = [os.path.abspath(root) for root, dirs, files in os.walk("./results") for name in files if name == "results.txt"]

    for folder in result_modalities:
        # Load data from results files
        train_data = np.loadtxt(os.path.join(folder, "train_perclass.txt"), delimiter=',')
        # result_data = np.loadtxt(os.path.join(folder, "results.txt"), delimiter=',')
        train_classes = np.unique(train_data[:,1])


        # Split per-class data into subarrays based on label
        train_data = np.split(train_data, np.unique(train_data[:,0], return_index=True)[1])[1:]
        # test_data = np.split(test_data, np.unique(test_data[:,0], return_index=True)[1])[1:]

        fig = plt.figure(figsize=(16,12))
        fig.suptitle(os.path.basename(folder) + " - " + title, fontsize=20, weight="bold")
        ax = fig.add_subplot(111)  
        ax.set_title(title, fontsize=20, weight="bold")
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        if (ylabel == "Accuracy"): 
            ax.set_ylim(0,1)

        ax.set_xlim(1, len(train_data))
        x = np.arange(1, len(train_data)+1)
        accuracies = np.empty((int(np.max(train_classes)+1), len(x)))
        accuracies[:] = np.nan
        
        for round in train_data:
            for label in round:
                accuracies[int(label[1])][int(label[0]-1)] = label[4]
        for y in accuracies:
            ax.plot(x, y,linewidth=2)

        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.legend(["class"+str(int(lbl)) for lbl in train_classes], loc="upper right")
        

        # Save each figure individually
        prefix, suffix = folder.split(os.sep + "results" + os.sep)
        fpath = os.path.join(prefix, "plots", suffix + " " + "Train Accuracy" + ".pdf")
        Path(os.path.dirname(fpath)).mkdir(parents=True, exist_ok=True)
        print("Saving", fpath)
        plt.savefig(fpath)
        plt.close(fig)

def Test_Accuracy_Per_Class():
    title =  "Per-Class Test Accuracy"
    xlabel = "Round"
    ylabel = "Accuracy"

    # Get paths to each parent folder containing result data
    result_modalities = [os.path.abspath(root) for root, dirs, files in os.walk("./results") for name in files if name == "results.txt"]

    for folder in result_modalities:
        # Load data from results files
        train_data = np.loadtxt(os.path.join(folder, "train_perclass.txt"), delimiter=',')
        test_data = np.loadtxt(os.path.join(folder, "test_perclass.txt"), delimiter=',')
        test_classes = np.unique(test_data[:,1])


        # Split per-class data into subarrays based on label
        train_data = np.split(train_data, np.unique(train_data[:,0], return_index=True)[1])[1:]
        test_data = np.split(test_data, np.unique(test_data[:,0], return_index=True)[1])[1:]

        # Generate individual plots

        fig = plt.figure(figsize=(18,12))
        fig.suptitle(os.path.basename(folder) + " - " + title, fontsize=20, weight="bold")
        ax1 = fig.add_subplot(111)  
        ax1.set_title(title, fontsize=14, weight="bold")
        ax1.set_xlabel(xlabel, fontsize=12)
        ax1.set_ylabel(ylabel, fontsize=12)
        if (ylabel == "Accuracy"): 
            ax1.set_ylim(-0.1,1.1)

            ax1.set_xlim(1, len(train_data))

            x = list(range(1, len(train_data), 2))
            accuracies = np.empty((int(np.max(test_classes)+1), len(x)))
            accuracies[:] = np.nan
            
            for round in test_data:
                for label in round:
                    accuracies[int(label[1])][int(label[0]/2)] = label[4]
            for y in accuracies:
                ax1.plot(x, y)

            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * .8, box.height])
            ax1.legend(["class"+str(int(lbl)) for lbl in test_classes], loc="center left", bbox_to_anchor=(1, 0.5))

        # Save each figure individually
        prefix, suffix = folder.split(os.sep + "results" + os.sep)
        fpath = os.path.join(prefix, "plots", suffix + " " + "Test Accuracy" + ".pdf")
        Path(os.path.dirname(fpath)).mkdir(parents=True, exist_ok=True)
        print("Saving", fpath)
        plt.savefig(fpath)
        plt.close(fig)

def Overall_Accuracy_Per_Class():
    title =  "Per-Class Overall Accuracy"
    xlabel = "Round"
    ylabel = "Accuracy"

    # Get paths to each parent folder containing result data
    result_modalities = [os.path.abspath(root) for root, dirs, files in os.walk("./results") for name in files if name == "results.txt"]

    for folder in result_modalities:
        # Load data from results files
        train_data = np.loadtxt(os.path.join(folder, "train_perclass.txt"), delimiter=',')
        test_data = np.loadtxt(os.path.join(folder, "test_perclass.txt"), delimiter=',')
        result_data = np.loadtxt(os.path.join(folder, "results.txt"), delimiter=',')

        # Split per-class data into subarrays based on label
        train_data = np.split(train_data, np.unique(train_data[:,0], return_index=True)[1])[1:]
        test_data = np.split(test_data, np.unique(test_data[:,0], return_index=True)[1])[1:]

        fig = plt.figure(figsize=(16,12))
        fig.suptitle(os.path.basename(folder) + " - " + title, fontsize=20, weight="bold")
        ax = fig.add_subplot(111)  
        ax.set_title(title, fontsize=20, weight="bold")
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        if (ylabel == "Accuracy"): 
            ax.set_ylim(0,1)

        ax.set_xlim(1, len(train_data))
        x = result_data[:,0]
        train_y = result_data[:,3]  # overall train accuracy
        test_y = result_data[:,5]  # overall test  accuracy
        ax.plot(x, train_y,linewidth=2)
        ax.plot(x, test_y,linewidth=2)
        ax.legend(["Training", "Testing"], loc="upper right")
        prefix, suffix = folder.split(os.sep + "results" + os.sep)
        fpath = os.path.join(prefix, "plots", suffix + " " + "Overall Accuracy" + ".pdf")
        Path(os.path.dirname(fpath)).mkdir(parents=True, exist_ok=True)
        print("Saving", fpath)
        plt.savefig(fpath)
        plt.close(fig)

def Class_Distribution():
    title =  "Class Distribution"

    # Get paths to each parent folder containing result data
    result_modalities = [os.path.abspath(root) for root, dirs, files in os.walk("./results") for name in files if name == "results.txt"]

    for folder in result_modalities:
        # Load data from results files
        fig = plt.figure(figsize=(16,12))
        fig.suptitle(os.path.basename(folder) + " - " + title, fontsize=20, weight="bold")
        ax = fig.add_subplot(111)  
        train_data = np.loadtxt(os.path.join(folder, "train_perclass.txt"), delimiter=',')
        test_data = np.loadtxt(os.path.join(folder, "test_perclass.txt"), delimiter=',')
        train_classes = np.unique(train_data[:,1])
        test_classes = np.unique(test_data[:,1])
        train_distribution, test_distribution = np.zeros(int(np.max(train_classes)+1)), np.zeros(int(np.max(test_classes)+1))

        # Split per-class data into subarrays based on label
        train_data = np.split(train_data, np.unique(train_data[:,0], return_index=True)[1])[1:]
        test_data = np.split(test_data, np.unique(test_data[:,0], return_index=True)[1])[1:]

        for round in train_data:
            for label in round:
                train_distribution[int(label[1])] += label[3]

        for round in test_data:
            for label in round:
                test_distribution[int(label[1])] += label[3]

        ax.set_xticks(np.arange(-1, len(train_distribution), 1))
        ax.set_yscale("log")
        x = np.arange(len(train_distribution))
        ax.bar(x+.2, train_distribution, .4)
        x = np.arange(len(test_distribution))
        ax.bar(x-.2, test_distribution, .4)

        ax.legend(["Training", "Testing"], loc="upper right")
        
        prefix, suffix = folder.split(os.sep + "results" + os.sep)
        fpath = os.path.join(prefix, "plots", suffix + " " + "Class Distribution" + ".pdf")
        Path(os.path.dirname(fpath)).mkdir(parents=True, exist_ok=True)
        print("Saving", fpath)
        plt.savefig(fpath)
        plt.close(fig)

def main():
    single_multi_modality_comparison()
    cross_modality_comparison()
    Train_Accuracy_Per_Class()
    Test_Accuracy_Per_Class()
    Overall_Accuracy_Per_Class()
    Class_Distribution()


if __name__ == "__main__":
    main()
 
