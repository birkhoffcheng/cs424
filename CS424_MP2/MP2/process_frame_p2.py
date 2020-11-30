from scheduling.misc import *
from scheduling.TaskEntity import *


# read the input cluster box data from file
box_info = read_json_file('../dataset/depth_clustering_detection_flat.json')

# Check if a point (x, y) is in a Box
def is_in(box, x, y):
	return box.left <= x and x <= box.right and box.up <= y and y <= box.down

class Box():
	def __init__(self, left, up, right, down, distance):
		self.left = left
		self.up = up
		self.right = right
		self.down = down
		self.distance = distance

	def is_in(self, x, y):
		return is_in(self, x, y)

	# Checks if two boxes intersect
	def intersects(self, target):
		close_prox = abs(self.distance - target.distance) <= 10
		target_in_self = self.is_in(target.left, target.up) or self.is_in(target.left, target.down) or self.is_in(target.right, target.up) or self.is_in(target.right, target.down)
		self_in_target = is_in(target, self.left, self.up) or is_in(target, self.left, self.down) or is_in(target, self.right, self.up) or is_in(target, self.right, self.down)
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

	clusters.sort(key=lambda x : (x.down - x.up) * (x.right - x.left))
	priority = 0
	task_batches = []
	for box in clusters:
		task = TaskEntity(image_path=frame.path, coord=[box.left, box.up, box.right, box.down], depth=box.distance)
		task_batch = TaskBatch([task], task.img_width, task.img_height, priority=priority)
		task_batches.append(task_batch)
		priority += 1

	return task_batches
