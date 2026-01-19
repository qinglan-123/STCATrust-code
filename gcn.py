import time
import torch
import csv
import numpy as np
from tqdm import trange
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F

from utils import calculate_auc
from convolution import GraphConvolutionalNetwork, AttentionNetwork,CrossAttentionNetwork,LSTMCrossAttention
from dataset import get_snapshot_index

# Define STCATrust class, inheriting from torch.nn.Module, used to build and train STCATrust model
class STCATrust(torch.nn.Module):
    # Class initialization method, receiving device, parameters, node features and label count as input
    def __init__(self, device, args, X, num_labels):
        # Call parent class initialization method
        super(STCATrust, self).__init__()
        # Save incoming parameters
        self.args = args
        # Set random seed to ensure reproducible results
        torch.manual_seed(self.args.seed)
        # Save device information
        self.device = device
        # Save node features
        self.X = X
        # Save dropout rate
        self.dropout = self.args.dropout
        # Save label count
        self.num_labels = num_labels
        # Build model's spatial and temporal layers
        self.build_model()
        # Initialize regression weights
        self.regression_weights = Parameter(torch.Tensor(self.args.layers[-1]*2, self.num_labels))
        # Use Xavier initialization method to initialize regression weights
        init.xavier_normal_(self.regression_weights)

    # Method to build model's spatial and temporal layers
    def build_model(self):
        """
        Constructing spatial and temporal layers.
        """
        # Build spatial layer using graph convolutional network
        self.structural_layer = GraphConvolutionalNetwork(self.device, self.args, self.X, self.num_labels)
        self.structural_layer0 = GraphConvolutionalNetwork(self.device, self.args, self.X, self.num_labels)
        self.structural_layer1 = GraphConvolutionalNetwork(self.device, self.args, self.X, self.num_labels)
        # Build temporal layer using attention network
        self.temporl_layer = LSTMCrossAttention(input_dim=self.args.layers[-1],n_heads=self.args.attention_head,num_time_slots=self.args.train_time_slots,attn_drop=0.5,residual=True)
        #self.temporl_layer = CrossAttentionNetwork(input_dim=self.args.layers[-1],n_heads=self.args.attention_head,num_time_slots=self.args.train_time_slots,attn_drop=0.5,residual=True)
        # self.temporl_layer = AttentionNetwork(input_dim=self.args.layers[-1], n_heads=self.args.attention_head,
        #                                            num_time_slots=self.args.train_time_slots, attn_drop=0.5,
        #                                            residual=True)#cross

    # Method to calculate loss function
    def calculate_loss_function(self, z, train_edges, target):
        """
        Calculating loss.
        :param z: Node embedding.
        :param train_edges: [2, #edges]
        :param target: Label vector storing 0 and 1.
        :return loss: Value of loss.
        """
        # Get embeddings of start and end nodes of training edges
        start_node, end_node = z[train_edges[0], :], z[train_edges[1], :]
        # Concatenate embeddings of start and end nodes
        features = torch.cat((start_node, end_node), 1)
        # Get prediction results through matrix multiplication
        predictions = torch.mm(features, self.regression_weights)

        # Handle imbalanced data, calculate class weights
        class_weight = torch.FloatTensor(1 / np.bincount(target.cpu()) * features.size(0))
        # Define cross entropy loss function and use class weights
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(self.device)
        # Calculate loss value
        loss_term = criterion(predictions, target)  # target corresponds to sorted_train_edges

        return loss_term

    # Forward propagation method
    def forward(self, train_edges, y, y_train, index_list):
        # Store output of spatial layer
        structural_out = []
        structural_out0 = []
        structural_out1 = []
        # Initialize indices
        index0 = 0
        index1 = 0
        index2 = 0
        # Iterate through each training time slot
        for i in range(self.args.train_time_slots):
            # Add spatial layer output of each time slot to the list
            structural_out.append(self.structural_layer(train_edges[:, index0:index_list[i]], y_train[index0:index_list[i], :]))
            structural_out0.append(
                self.structural_layer(train_edges[:, index0:index_list[i]], y_train[index0:index_list[i], :]))
            structural_out1.append(
                self.structural_layer(train_edges[:, index0:index_list[i]], y_train[index0:index_list[i], :]))
            # Update indices
            index0 = index_list[i]
            index1 = index_list[i]
            index2 = index_list[i]

        # Stack outputs of spatial layer
        structural_out = torch.stack(structural_out)
        structural_out0 = torch.stack(structural_out0)
        structural_out1 = torch.stack(structural_out1)
        # Adjust dimension order
        structural_out = structural_out.permute(1,0,2)  # [N,T,F] [5881,7,32]
        structural_out0 = structural_out0.permute(1,0,2)
        structural_out1 = structural_out1.permute(1,0,2)
        #print(structural_out, structural_out0, structural_out1)
        # Get temporal layer output through temporal layer
        temporal_all = self.temporl_layer(structural_out0,structural_out1)  # [N,T,F](cross)
        #temporal_all = self.temporl_layer(structural_out)  # [N,T,F]
        # Get final embedding of target time slot
        temporal_out = temporal_all[:, self.args.train_time_slots-1, :].squeeze()  # [N,F]
        # Calculate loss value
        loss = self.calculate_loss_function(temporal_out, train_edges, y)

        return loss, temporal_out

# Define GCNTrainer class for training and evaluating STCATrust model
class GCNTrainer(object):
    """
    Object to train and score the STCATrust, log the model behaviour and save the output.
    """
    # Class initialization method, receiving parameters and edge data as input
    def __init__(self, args, edges):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param edges: Edge data structure.
        """
        # Save incoming parameters
        self.args = args
        # Save edge data
        self.edges = edges
        # Select device (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        # Record global start time
        self.global_start_time = time.time()
        # Setup logs
        self.setup_logs()

    # Method to setup logs
    def setup_logs(self):
        """
        Creating a log dictionary for recording performance.
        """
        # Initialize log dictionary
        self.logs = {}
        # Save parameter information
        self.logs["parameters"] = vars(self.args)
        # Initialize performance logs
        self.logs["performance"] = [["Epoch", "MCC", "AUC", "ACC_Balanced", "AP", "F1_Micro", "F1_Macro"]]
        # Initialize training time logs
        self.logs["training_time"] = [["Epoch", "Seconds"]]

    # Method to setup dataset
    def setup_dataset(self):
        """
        Creating training snapshots and testing snapshots.
        """
        # Get index of each time slot
        self.index_list = get_snapshot_index(self.args.time_slots, data_path=self.args.data_path)
        # Get index of current training time slot
        index_t = self.index_list[self.args.train_time_slots-1]

        print("--------------- Getting training and testing snapshots starts ---------------")
        print('Snapshot index',self.index_list,index_t)  # index_t denotes the index of snapshot t
        # Get training edges
        self.train_edges = self.edges['edges'][:index_t]
        # Get training labels
        self.y_train = self.edges['labels'][:index_t]
        # Get node set in training set
        train_set = set(list(self.train_edges.flatten()))

        # Single time slot prediction
        if self.args.single_prediction:
            # index_t_1 = self.index_list[self.args.train_time_slots-2]  # index of snapshot t-1, used for single-timeslot prediction on unobserved nodes, i.e., task 3
            # train_pre = set(list(self.train_edges[:index_t_1].flatten()))  # for task 3
            # train_t = set(list(self.train_edges[index_t_1:].flatten()))  # for task 3

            # Get index of next time slot
            index_t1 = self.index_list[self.args.train_time_slots]  # index of snapshot t+1
            # Get test edges
            self.test_edges = self.edges['edges'][index_t:index_t1]
            # Get test labels
            self.y_test = self.edges['labels'][index_t:index_t1]
            print('{} edges at snapshot t+1'.format(len(self.test_edges)))

            # Store edges of observed nodes
            self.obs = []
            # Store test labels of observed nodes
            self.y_test_obs = []
            # self.unobs = []  # for task 3
            # self.y_test_unobs = []  # for task 3
            # Iterate through test edges
            for i in range(len(self.test_edges)):
                # Get start and end nodes of the edge
                tr = self.test_edges[i][0]
                te = self.test_edges[i][1]
                # If both start and end nodes are in training set, add to observed edges list
                if tr in train_set and te in train_set:
                    self.obs.append(self.test_edges[i])
                    self.y_test_obs.append(self.y_test[i])
                # for task 3
                # if tr in train_t and tr not in train_pre and te not in train_t and te in train_pre:
                #     self.unobs.append(self.test_edges[i])
                #     self.y_test_unobs.append(self.y_test[i])
                # elif te in train_t and te not in train_pre and tr not in train_t and tr in train_pre:
                #     self.unobs.append(self.test_edges[i])
                #     self.y_test_unobs.append(self.y_test[i])
                # elif tr in train_t and te in train_t and tr not in train_pre and te not in train_pre:
                #     self.unobs.append(self.test_edges[i])
                #     self.y_test_unobs.append(self.y_test[i])

            # Initialize positive sample count
            self.pos_count = 0
            # Initialize negative sample count
            self.neg_count = 0
            # Iterate through observed test labels
            for i in range(len(self.y_test_obs)):
                # If label is positive sample, increment positive sample count
                if self.y_test_obs[i][0] == 1:
                    self.pos_count += 1
                else:
                    # Otherwise increment negative sample count
                    self.neg_count += 1
            print('Trust and distrust distribution:',self.pos_count,self.neg_count)
            # print('Observed single-timeslot test edges'.format(len(self.unobs)))  # for task 3
            print('Observed single-timeslot test edges:',len(self.obs))

            # Convert observed edges to numpy array
            self.obs = np.array(self.obs)
            # Convert observed test labels to numpy array
            self.y_test_obs = np.array(self.y_test_obs)
            # self.unobs = np.array(self.unobs)  # for task 3
            # self.y_test_unobs = np.array(self.y_test_unobs)  # for task 3
        else:   # Multi time slot prediction
            # Get index of current training time slot
            index_pre = self.index_list[self.args.train_time_slots - 1]
            # Get index of next three time slots
            index_lat = self.index_list[self.args.train_time_slots + 2]
            # Get test edges
            self.test_edges = self.edges['edges'][index_pre:index_lat]
            # Get test labels
            self.y_test = self.edges['labels'][index_pre:index_lat]
            print('{} edges from snapshot t+1 to snapshot t+3'.format(len(self.test_edges)))

            # Store edges of observed nodes
            self.obs = []
            # Store test labels of observed nodes
            self.y_test_obs = []

            # Iterate through test edges
            for i in range(len(self.test_edges)):
                # If both start and end nodes are in training set, add to observed edges list
                if self.test_edges[i][0] in train_set and self.test_edges[i][1] in train_set:
                    self.obs.append(self.test_edges[i])
                    self.y_test_obs.append(self.y_test[i])

            # Initialize positive sample count
            self.pos_count = 0
            # Initialize negative sample count
            self.neg_count = 0
            # Iterate through observed test labels
            for i in range(len(self.y_test_obs)):
                # If label is positive sample, increment positive sample count
                if self.y_test_obs[i][0] == 1:
                    self.pos_count += 1
                else:
                    # Otherwise increment negative sample count
                    self.neg_count += 1
            print('Trust and distrust distribution:',self.pos_count,self.neg_count)
            print('Observed multi-timeslot test edges:',len(self.obs))

            # Convert observed edges to numpy array
            self.obs = np.array(self.obs)
            # Convert observed test labels to numpy array
            self.y_test_obs = np.array(self.y_test_obs)
        print("--------------- Getting training and testing snapshots ends ---------------")

        # Node features should be loaded from actual dataset
        # Placeholder for actual feature loading - this would need to be implemented based on the specific dataset structure
        # For now, we'll use a placeholder that indicates features need to be loaded properly
        raise NotImplementedError("Node features must be loaded from actual dataset")
        
        # Get label count
        self.num_labels = np.shape(self.y_train)[1]

        # Convert training labels to torch tensor
        self.y = torch.from_numpy(self.y_train[:,1]).type(torch.long).to(self.device)
        # convert vector to number 0/1, 0 represents trust and 1 represents distrust

        # Convert training edges to torch tensor
        self.train_edges = torch.from_numpy(np.array(self.train_edges, dtype=np.int64).T).type(torch.long).to(self.device)  # (2, #edges)
        # Convert training labels to torch tensor
        self.y_train = torch.from_numpy(np.array(self.y_train, dtype=np.float32)).type(torch.float).to(self.device)
        # Convert label count to torch tensor
        self.num_labels = torch.from_numpy(np.array(self.num_labels, dtype=np.int64)).type(torch.long).to(self.device)
        # Convert node features to torch tensor
        self.X = torch.from_numpy(self.X).to(self.device)

    # Method to create and train the model
    def create_and_train_model(self):
        """
        Model training and scoring.
        """
        print("\nTraining started.\n")
        learning_rates = [0.005]
        all_losses = []  # Store loss values for all learning rates

        for lr in learning_rates:
            self.model = STCATrust(self.device, self.args, self.X, self.num_labels).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)
            self.model.train()
            epochs = trange(self.args.epochs, desc="Loss")
            losses = []  # Store loss values for current learning rate

            for epoch in epochs:
                print("Epoch:", epoch)
                self.model.train()
                self.optimizer.zero_grad()
                start_time = time.time()
                loss, final_embedding = self.model(self.train_edges, self.y, self.y_train, self.index_list)
                loss.backward()
                epochs.set_description("STCATrust (Loss=%g)" % round(loss.item(), 4))
                self.optimizer.step()
                self.score_model(epoch)
                self.logs["training_time"].append([epoch + 1, time.time() - start_time])
                self.score_model(epoch)
                losses.append(loss.item())

            all_losses.append(losses)

        # Save loss values to CSV file
        with open('loss.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(learning_rates)  # Write learning rates as column names
            for row in zip(*all_losses):
                writer.writerow(row)

        self.logs["training_time"].append(["Total", time.time() - self.global_start_time])
    # Method to evaluate the model
    def score_model(self, epoch):
        """
        Score the model on the test set edges in each epoch.
        :param epoch: Epoch number.
        """
        # Set model to evaluation mode
        self.model.eval()
        # Forward propagation, calculate loss and training embedding
        loss, self.train_z = self.model(self.train_edges, self.y, self.y_train, self.index_list)
        # Convert observed edges to torch tensor
        score_edges = torch.from_numpy(np.array(self.obs, dtype=np.int64).T).type(torch.long).to(self.device)
        # Concatenate training embeddings
        test_z = torch.cat((self.train_z[score_edges[0, :], :], self.train_z[score_edges[1, :], :]), 1)
        # score_edges[0, :] is the index of trustors, while score_edges[1, :] is the index of trustees
        # Get scores through matrix multiplication
        scores = torch.mm(test_z, self.model.regression_weights.to(self.device))

        # Apply softmax to scores to get prediction probabilities
        predictions = F.softmax(scores, dim=1)

        # Calculate evaluation metrics
        mcc, auc, acc_balanced, precision, f1_micro, f1_macro = calculate_auc(predictions, self.y_test_obs)
        print('mcc, auc, acc_balanced, precision, f1_micro, f1_macro \n')
        print('%.4f' % mcc, '%.4f' % auc, '%.4f' % acc_balanced, '%.4f' % precision, '%.4f' % f1_micro, '%.4f' % f1_macro)

        # Record evaluation metrics
        self.logs["performance"].append([epoch + 1, mcc, auc, acc_balanced, precision, f1_micro, f1_macro])