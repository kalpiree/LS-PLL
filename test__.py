import torch
import torch.nn.functional as F

# Tensor with elements 0 or 1
# tensor_zeros_ones = torch.randint(0, 2, (2, 10))
#
# # Tensor with random numbers
# tensor_random = torch.rand(2, 10)
#
# print("Tensor with 0s and 1s:")
# print(tensor_zeros_ones)
#
# print("\nTensor with random numbers:")
# print(tensor_random)

import torch


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

        # Choose a random index from label_set_indices
        random_index = torch.randint(len(label_set_indices), size=(1,))
        ground_truth_label_index = label_set_indices[random_index].item()

        # Store the results for this datapoint
        ground_truth_labels[i] = ground_truth_label_index
        label_set_indices_list.append(label_set_indices)

    # Convert the list of tensors to a single tensor
    # label_set_indices_tensor = torch.stack(label_set_indices_list)

    return ground_truth_labels, label_set


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


def calculate_weighted_term_mean(label_set, outputs):
    num_datapoints = label_set.shape[0]

    # Initialize lists to store intermediate results for each datapoint
    fij_terms = []
    weighted_terms = []
    loss = 0

    for datapoint in range(num_datapoints):
        # Step 1: Extract Indices of Label Set
        label_set_indices = (label_set[datapoint] != 0).nonzero(as_tuple=True)[0]
        #print(label_set_indices)
        # Step 2: Calculate fij Term
        output_probs = F.softmax(outputs[datapoint, :], dim=0)
        fij_term = torch.log(torch.sum(torch.exp(output_probs)))
        #print("fij_term->",fij_term)
        fij_terms.append(fij_term.item())
        
        # Step 3: Calculate Weighted Term weighted_term = label_set[datapoint, label_set_indices] * (torch.sum(
        # outputs[datapoint, label_set_indices]) - fij_term)
        weighted_term = label_set[datapoint, label_set_indices] * ((output_probs[label_set_indices]) - fij_term)
        #print(weighted_term)
        sum_result = torch.sum(-1 * weighted_term)
        #print("sum_result->",sum_result)
        # weighted_terms.append(weighted_term.item())
        # print("\nStep 3: Calculate Weighted Term")
        #print("weighted_sum =", sum_result)
        loss += sum_result
        #print("loss",loss)
        #print("loss:", loss)

    # Step 4: Calculate Sum for Each Datapoint and Sum Across All Datapoints
    # sum_per_datapoint = torch.tensor(weighted_terms)
    # sum_across_datapoints = torch.sum(sum_per_datapoint)

    # Calculate the mean
    mean_weighted_term = loss / num_datapoints
    print("Weighted mean loss->",mean_weighted_term)
    # print("\nStep 4: Calculate Sum for Each Datapoint and Sum Across All Datapoints")
    # print("sum_per_datapoint =", sum_per_datapoint.tolist())
    # print("sum_across_datapoints =", sum_across_datapoints.item())

    # Final Result
    #print("\nFinal Result (Mean):", mean_weighted_term)

    return mean_weighted_term


# # Example usage with the provided data
# label_set = torch.tensor([[0.9, 0, -0.7, 0, 0.6], [0.87, 1, 0, 0, 0]])
# outputs = torch.tensor([[0.5, 1.2, 0.8, 1.6, 2.2], [1.0, 0.7, 0.3, 0.5, 0.6]])


# softmax_accumulator = torch.zeros(outputs.shape)


def update_ground_truth_labels(outputs, label_sets, softmax_accumulator, weighting_parameter=0.1):
    # Initialize accumulator if not provided
    if softmax_accumulator is None:
        softmax_accumulator = torch.zeros(outputs.shape)

    # Initialize normalized softmax
    normalized_softmax = torch.zeros(outputs.shape)

    # Initialize updated ground-truth labels
    updated_ground_truth_labels = torch.zeros(len(label_sets), dtype=torch.long)

    for i in range(len(label_sets)):
        #print("Datapoint->", i)
        #print("Label_set->", i, label_sets[i])
        # Identify column indices where the value is non-zero in outputs
        non_zero_indices = torch.nonzero(label_sets[i, :]).squeeze()
        #print("Non-zero-indices->", non_zero_indices)

        # Calculate softmax according to Equation (17)
        softmax_output = torch.exp(outputs[i, :]) / torch.exp(outputs[i, non_zero_indices]).sum()
        #print("Softmax_output->", softmax_output)

        # Accumulate softmax scores with moving average
        #softmax_accumulator[i, :] = weighting_parameter * softmax_accumulator[i, :] + (
        #       1 - weighting_parameter) * softmax_output
        #print('softmax_accumulator->', softmax_accumulator[i])
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

        # softmax_accumulator[i, :] = torch.where(non_zero_indices > 0, softmax_accumulator[i, :] / softmax_accumulator[i, non_zero_indices].sum(), 0.0)
        # print('softmax_accumulator->', softmax_accumulator[i])

        # Update ground-truth label via Eq. (20)
        updated_ground_truth_labels[i] = torch.argmax(softmax_accumulator_hat[i, :])

        # print("Updated_ground_truth_labels->", updated_ground_truth_labels[i])

        # normalized_softmax[i, :] = softmax_accumulator[i, :]

    return updated_ground_truth_labels, softmax_accumulator
