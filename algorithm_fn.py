import torch
import torch.nn.functional as F
import torch


## function for randomly choosing a ground truth from the label set

def initialize_ground_truth_labels(label_set):
    # Get the number of datapoints
    num_datapoints = label_set.size(0)

    # Initialize empty tensors to store ground truth labels and indices for each datapoint
    ground_truth_labels = torch.zeros(num_datapoints, dtype=torch.int64)
    label_set_indices_list = []

    # Iterate over each datapoint
    for i in range(num_datapoints):
        # Get indices of the label set (where label is 1)
        label_set_indices = label_set[i].nonzero().squeeze()
        # Handle the scalar case, for example, convert it to a 1-dimensional tensor
        if label_set_indices.dim() == 0:
            label_set_indices = torch.tensor([label_set_indices.item()])
        
        # Choose a random index from label_set_indices
        random_index = torch.randint(len(label_set_indices), size=(1,))
        ground_truth_label_index = label_set_indices[random_index].item()

        # Store the results for this datapoint
        ground_truth_labels[i] = ground_truth_label_index
        label_set_indices_list.append(label_set_indices)


    return ground_truth_labels, label_set



## function for smoothing and updating values in the label set
def update_smoothed_labels(label_set, ground_truth_labels, smoothing_rate):
    updated_label_set = torch.zeros_like(label_set, dtype=torch.float64)
    for i in range(len(ground_truth_labels)):
        ground_truth_label_index = ground_truth_labels[i]

        for j in range(len(label_set[i])):
            if j == ground_truth_label_index:
                updated_label_set[i, j] = float(
                    (1 - smoothing_rate) + smoothing_rate / (label_set[i, :] != 0).sum().item())
            elif label_set[i, j] != 0:
                updated_label_set[i, j] = float(smoothing_rate / (label_set[i, :] != 0).sum().item())
    return updated_label_set



## function for calculating losses
def calculate_weighted_term_mean(label_set, outputs):
    num_datapoints = label_set.shape[0]

    # Initialize lists to store intermediate results for each datapoint
    fij_terms = []
    weighted_terms = []
    loss = 0

    for datapoint in range(num_datapoints):
        # Extract Indices of Label Set
        label_set_indices = (label_set[datapoint] != 0).nonzero(as_tuple=True)[0]
        # Calculate fij Term
        output_probs = F.softmax(outputs[datapoint, :], dim=0)
        fij_term = torch.log(torch.sum(torch.exp(output_probs)))
        fij_terms.append(fij_term.item())
        
        # Calculate Weighted Term
        weighted_term = label_set[datapoint, label_set_indices] * ((output_probs[label_set_indices]) - fij_term)
        sum_result = torch.sum(-1 * weighted_term)
        loss += sum_result

    # Calculate Sum for Each Datapoint and Sum Across All Datapoints
    mean_weighted_term = loss / num_datapoints
    return mean_weighted_term


## function for updating the ground truth after every iteration
def update_ground_truth_labels(outputs, label_sets, softmax_accumulator, weighting_parameter=0.1):
    # Initialize accumulator if not provided
    if softmax_accumulator is None:
        softmax_accumulator = torch.zeros(outputs.shape)

    # Initialize normalized softmax
    normalized_softmax = torch.zeros(outputs.shape)

    # Initialize updated ground-truth labels
    updated_ground_truth_labels = torch.zeros(len(label_sets), dtype=torch.long)

    for i in range(len(label_sets)):
        non_zero_indices = torch.nonzero(label_sets[i, :]).squeeze()
        softmax_output = torch.exp(outputs[i, :]) / torch.exp(outputs[i, non_zero_indices]).sum()
        new_values = weighting_parameter * softmax_accumulator[i, :] + (
            1 - weighting_parameter) * softmax_output
        softmax_accumulator[i, :] = new_values

        softmax_accumulator_hat = torch.zeros(softmax_accumulator.shape)
        # Normalize the accumulated softmax score according to Equation (19)
        for j in range(len(label_sets[i])):
            if label_sets[i][j] != 0:
                softmax_accumulator_hat[i, j] = softmax_accumulator[i, j] / softmax_accumulator[
                    i, non_zero_indices].sum()
            else:
                softmax_accumulator_hat[i, j] = 0

        # Update ground-truth label via Eq. (20)
        updated_ground_truth_labels[i] = torch.argmax(softmax_accumulator_hat[i, :])

    return updated_ground_truth_labels, softmax_accumulator
