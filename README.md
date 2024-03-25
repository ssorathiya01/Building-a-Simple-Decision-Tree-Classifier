# Building a Simple Decision Tree Classifierr
Introduction:
In the machine learning world, decision trees are powerful tools for classification tasks. In this blog post, we will explore how to build a decision tree classifier using Python. Using a simple example, we classify mushrooms as edible or poisonous based on their physiological properties.
Understanding the Problem:
Imagine starting a business that involves selling wild mushrooms. Because not all mushrooms are safe, a way to distinguish between edible ones is needed. To solve this problem, we use data that contain information on scale factors such as cap color, tree size, and whether it grows solitary or not. Our dataset contains 10 mushroom models, each defined by three factors: cap color (brown or red), trunk shape (thin or growing), solitary growth (yes or no). In addition, we have a score indicating whether each mushroom is edible (1) or poisonous (0). To simplify our analysis, we hot-encoded one of the items, converting it to 0 or 1 values.
Building the Decision Tree:
Calculate Entropy:
def compute_entropy(y):
    entropy = 0.
    if len(y) != 0:
        p1 = len(y[y == 1]) / len(y) 
        if p1 != 0 and p1 != 1:
            entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
        else:
            entropy = 0
    return entropy
Split the Dataset:
def split_dataset(X, node_indices, feature):
    left_indices = []
    right_indices = []
    for i in node_indices:   
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices
Calculate Information Gain:
def compute_information_gain(X, y, node_indices, feature):
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)
    weighted_entropy = w_left * left_entropy + w_right * right_entropy
    information_gain = node_entropy - weighted_entropy
    return information_gain
Get the Best Split:
def get_best_split(X, y, node_indices):   
    num_features = X.shape[1]
    best_feature = -1
    max_information_gain=0
    for i in range(num_features):
        information_gain = compute_information_gain(X, y, node_indices, i)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_feature = i
    return best_feature
Building the Tree:
Using the various functions we created, we create our decision tree by repeatedly dividing the dataset into left and right branches based on our choices. We visualize a tree to better understand how it makes decisions.
tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
    
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)

build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
Conclusion:
In this blog post, we took the approach of building a simple decision tree classifier from scratch. Decision trees provide a transparent and interpretable way to classify data, making them useful in a variety of applications. By understanding the basics of building decision trees you can gain insight into how machine learning systems make decisions and apply this knowledge to solve real-world problems.
