from scheduling.misc import *
from scheduling.TaskEntity import *


# read the input cluster box data from file
box_info = read_json_file('../dataset/depth_clustering_detection_flat.json')

class Box():
	def __init__(self, left, up, right, down, distance):
		self.left = left
		self.up = up
		self.right = right
		self.down = down
		self.distance = distance

	def is_in(self, x, y):
		return self.left <= x and x <= self.right and self.up <= y and y <= self.down

	# Checks if two boxes intersect
	def intersects(self, target):
		close_prox = abs(self.distance - target.distance) <= 7
		target_in_self = self.is_in(target.left, target.up) or self.is_in(target.left, target.down) or self.is_in(target.right, target.up) or self.is_in(target.right, target.down)
		self_in_target = target.is_in(self.left, self.up) or target.is_in(self.left, self.down) or target.is_in(self.right, self.up) or target.is_in(self.right, self.down)
		return close_prox and (target_in_self or self_in_target)

	# Union two boxes into a bigger box
	def union(self, target):
		self.left = min(self.left, target.left)
		self.up = min(self.up, target.up)
		self.right = max(self.right, target.right)
		self.down = max(self.down, target.down)
		self.distance = min(self.distance, target.distance)

def process_frame(frame):
	"""Process frame for scheduling.

	Process a image frame to obtain cluster boxes and corresponding scheduling parameters
	for scheduling.

	Student's code here.

	Args:
		param1: The image frame to be processed.

	Returns:
		A list of task_batches with each task_batch containing some tasks.
	"""

	cluster_boxes_data = get_cluster_box_info(frame, box_info)

	# student's code here
	clusters = []
	for box in cluster_boxes_data:
		processed = False
		target = Box(box[0], box[1], box[2], box[3], box[4])
		for cluster in clusters:
			if cluster.intersects(target):
				cluster.union(target)
				processed = True
				break
		if not processed:
			clusters.append(target)

	# Sort the clusters by area in reverse order so overlap removal always removes the small boxes
	clusters.sort(key=lambda x : (x.down - x.up) * (x.right - x.left), reverse=True)
	to_remove = []
	for i in range(len(clusters)):
		if clusters[i] in to_remove:
			continue
		for j in range(i+1, len(clusters)):
			if clusters[i].intersects(clusters[j]):
				clusters[i].union(clusters[j])
				if clusters[j] not in to_remove:
					to_remove.append(clusters[j])

	for i in to_remove:
		clusters.remove(i)

	clusters.sort(key=lambda x : x.distance)
	priority = 0
	task_batches = []
	for box in clusters:
		task = TaskEntity(image_path=frame.path, coord=[box.left, box.up, box.right, box.down], depth=box.distance)
		task_batch = TaskBatch([task], task.img_width, task.img_height, priority=priority)
		task_batches.append(task_batch)
		priority += 1

	return task_batches

# deadline miss rate is:  0.0
# [16, 43, 46, 51, 55, 57, 50, 51, 0, 0]
# average coverage: 0.723
# average accuracy: 0.394
