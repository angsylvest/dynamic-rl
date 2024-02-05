import numpy as np

class UltrasonicSensor:
    def __init__(self, num_rays, max_range):
        self.num_rays = num_rays
        self.max_range = max_range

    def detect_distance(self, agent_position, obstacle_positions):
        # Convert inputs to NumPy arrays for vectorized calculations
        agent_position = np.array(agent_position)
        obstacle_positions = np.array(obstacle_positions)

        # Initialize variables to store minimum distance and the angle of the ray
        min_distance = np.inf
        min_angle = None

        # Calculate angles for all rays
        angles = np.linspace(0, 2 * np.pi, self.num_rays, endpoint=False)

        # Calculate direction vectors for all rays
        direction_vectors = np.column_stack((np.cos(angles), np.sin(angles)))

        # Cast all rays from the agent position
        ray_ends = agent_position + direction_vectors * self.max_range

        # Broadcasted intersection check with all obstacles
        intersections = self.intersect(agent_position, ray_ends[:, np.newaxis, :], obstacle_positions)

        # Find the minimum distance and corresponding angle
        min_distances = np.linalg.norm(intersections - agent_position, axis=2)
        min_distance_index = np.argmin(min_distances)
        min_distance = min_distances.flat[min_distance_index]
        min_angle = angles[min_distance_index]

        return min_distance, min_angle

    def intersect(self, start, end, obstacle_positions):
        # Bounding box check with broadcasting
        min_x = np.minimum(start[0], end[0])
        max_x = np.maximum(start[0], end[0])
        min_y = np.minimum(start[1], end[1])
        max_y = np.maximum(start[1], end[1])

        obstacle_x = obstacle_positions[:, 0]
        obstacle_y = obstacle_positions[:, 1]

        return np.logical_and(min_x <= obstacle_x, obstacle_x <= max_x) & \
               np.logical_and(min_y <= obstacle_y, obstacle_y <= max_y)

# Example usage
sensor = UltrasonicSensor(num_rays=8, max_range=10)
agent_position = (0, 0)
obstacle_positions = np.array([(3, 2), (1, 1)])

distance, angle = sensor.detect_distance(agent_position, obstacle_positions)
print(f"Minimum distance: {distance}, Angle: {angle}")