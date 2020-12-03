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

class Cluster():
    def __init__(self, task, width, height, distance):
        self.tasks = [task]
        self.orig_width = width
        self.orig_height = height
        self.max_width = width
        self.max_height = height
        self.min_dist = distance


    def can_add_task(self, task, width, height):
        #If new width and height within 10% of original 
        #if width <= 10+self.orig_width and height <= 10+self.orig_height and width >= self.orig_width-10 and height >= self.orig_height-10:
        abs_tol = 300
        if self.orig_width*0.8 < abs_tol or self.orig_width*0.8 < abs_tol:
            return False
        if width <= 1.8*self.orig_width and height <= 1.8*self.orig_height:
            #If new task close in dimension
            return True
        return False    


    def add_task(self, task, width, height, distance):
        self.tasks.append(task)
        if width > self.max_width:
            self.max_width = width
        if height > self.max_height:
            self.max_height = height
        if distance < self.min_dist:
            self.min_dist = distance  
             
    def make_same_size(self):
        #box[0] = top left x
        #box[1] = top left y
        #box[2] = bottom right x
        #box[3] = bottom right y
        for box in self.tasks:
            #   what width should be   what width is
            xdiff = self.max_width - (box.right - box.left)
            if xdiff > 0:
                box.right += xdiff
                if box.right > 1919: #if pixel count is past edge of image expand other end
                    x_rem = box.right - 1919
                    box.right = 1919
                    box.left -= x_rem
            ydiff = self.max_height - (box.down - box.up)
            if ydiff > 0:
                box.down += ydiff
                if box.down > 1279: #if box goes past edge expand on other side
                    y_rem = box.down - 1279
                    box.down = 1279
                    box.up -= y_rem
   
    
    def get_tasks(self):
        return self.tasks

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


    # student's code here
    batched_clusters = []
    #sort boxes by smallest area to largest
    clusters.sort(key=lambda x: (x.right-x.left)*(x.down-x.up))
    for box in clusters:
        width = box.right-box.left
        height = box.down-box.up
        added = False
        for clstr in batched_clusters:
            if clstr.can_add_task(box, width, height):
                clstr.add_task(box, width, height, box.distance)
                added = True
                break
        if not added:
            new_cluster = Cluster(box, width, height, box.distance)
            batched_clusters.append(new_cluster)
    
    #batch tasks
    priority = 0
    task_batches = []
    batched_clusters.sort(key=lambda x : x.min_dist)
    for clstr in batched_clusters:
        #make all tasks images same dimensions 
        clstr.make_same_size()
        task_list = clstr.get_tasks()
        batch_list = []
        #make the batch list
        for box in task_list:
            next_task = TaskEntity(image_path=frame.path, coord=[box.left, box.up, box.right, box.down], depth = box.distance)
            batch_list.append(next_task)
        #add to return list
        task_batch = TaskBatch(batch_list, batch_list[0].img_width, batch_list[0].img_height, priority = priority)
        task_batches.append(task_batch)
        priority += 1


    return task_batches

# deadline miss rate is:  0.0
# [16, 43, 46, 51, 55, 57, 50, 51, 0, 0]
# average coverage: 0.723
# average accuracy: 0.394