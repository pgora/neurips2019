# This file contains code for generating toy datasets to fit using neural networks

def generate_dataset_based_on_graph(graph, nb_of_points):
    points = generate_points(size=graph.shape[0], nb_of_points=nb_of_points)
    values = get_values_for_points(points, graph)
    return points, values


def generate_points(size, nb_of_points):
    return np.random.randint(
        low=0, 
        high=120, 
        size=(nb_of_points, size),
    )


def get_values_for_points(points, graph):
    return np.array([get_value_for_point(point, graph) for point in points])


def get_value_for_point(point, graph):
    full_entropy = 0
    rescaled_point = point / 40.0
    for i in range(graph.shape[0]):
        row_sum = 0
        row_ent = 0
        for j in range(graph.shape[1]):
            if graph[i, j] == 0:
                continue
            current_distance = 0.5 * (point[i] - point[j]) ** 2
            exp_current_distance = np.exp(-current_distance)
            row_ent += current_distance * exp_current_distance
            row_sum += exp_current_distance
        if row_sum > 0:
            full_entropy += row_ent / row_sum + np.log(row_sum)
    return full_entropy