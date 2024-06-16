# colab:
https://colab.research.google.com/drive/181Ilic3WX4vtJJ2slFANs6cUB23oatwB?usp=sharing

# Exercise 1
Error: The provided function id_to_fruit is intended to return a fruit from a set based on a specified index. However, the function does not work correctly because it relies on the order of elements in a set, which is inherently unordered in Python. This means that the order of elements in a set can vary between executions, leading to inconsistent and incorrect results.
Symptoms: When running the function with a set of fruits, the returned fruit may not match the expected fruit at the given index. For example, given the set {"apple", "orange", "melon", "kiwi", "strawberry"} and requesting the fruit at index 1, the function might return 'melon' instead of the expected 'orange'.
Solution: To ensure consistent and predictable results, we need to convert the set to a list and sort it. This will give us a predictable order that we can rely on to fetch the correct fruit at a specified index. 

# Exercise 2
Obvious Error: The line where the coordinates are swapped contains an incorrect assignment, resulting in incorrect flipping of x and y coordinates. The current code assigns coords[:, 1] to both coords[:, 0] and coords[:, 1], leading to data loss and incorrect coordinates.
Fixing the Obvious Error: To properly swap the x and y coordinates, we need to use a temporary variable or perform the swaps in a way that doesn't overwrite values prematurely.

# Exercise 3
For some reason, the plot is not showing correctly. Can you find out what is going wrong? The current implementation of the plot_data function reads the data from the CSV file, but it does not convert the read strings into numerical values. As a result, the plot does not display correctly because it tries to plot strings rather than numbers.
How could this be fixed? To fix this, we need to ensure that the data read from the CSV file is converted to numerical values (i.e., floats). Additionally, it's important to plot the data with precision on the x-axis and recall on the y-axis, as specified.
Explanation: Convert Values to Float:
The values read from the CSV file are initially strings. The list comprehension [float(value) for value in row] converts each value in the row from a string to a float, ensuring the data can be plotted correctly. Plot Correct Data:
The plt.plot(results[:, 0], results[:, 1]) line plots precision on the x-axis and recall on the y-axis as specified. Additional Plot Customization:
Added plt.title('Precision-Recall Curve') for clarity. Enabled grid lines with plt.grid(True) to make the plot easier to read. Testing the Function: The test code creates a CSV file with sample data and then calls the plot_data function to generate the plot. This ensures that the data is read and plotted correctly, producing the expected precision-recall curve.


# Exercise 4
The provided train_gan function has a structural bug that is triggered when the batch size is changed from 32 to 64, and a cosmetic bug.

Structural Bug Problem: The error message "Using a target size (torch.Size([128, 1])) that is different to the input size (torch.Size([96, 1])) is deprecated. Please ensure they have the same size." indicates that the sizes of the concatenated tensors real_samples and generated_samples are not consistent with the batch size, leading to mismatched dimensions during training.
Solution: Ensure that the size of the tensors being concatenated and used for loss calculation match the batch size. The current code assumes the batch size is fixed for all iterations, but the last batch might be smaller. We need to dynamically get the size of the current batch.

Cosmetic Bug Problem: The function always titles generated sample figures as "Generate images\n Epoch: ...", which is slightly redundant and not as informative as it could be.
Solution: Improve the naming convention for clarity and simplicity.
