from scheduling.misc import *
from scheduling.TaskEntity import *


# read the input cluster box data from file
box_info = read_json_file('../dataset/depth_clustering_detection_flat.json')

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
            xdiff = self.max_width - (box[2] - box[0])
            if xdiff > 0:
                box[2] += xdiff
                if box[2] > 1919: #if pixel count is past edge of image expand other end
                    x_rem = box[2] - 1919
                    box[2] = 1919
                    box[0] -= x_rem
            ydiff = self.max_height - (box[3] - box[1])
            if ydiff > 0:
                box[3] += ydiff
                if box[3] > 1279: #if box goes past edge expand on other side
                    y_rem = box[3] - 1279
                    box[3] = 1279
                    box[1] -= y_rem
   
    
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

    task_batches = []

    # student's code here
    clusters = []
    #sort boxes by smallest area to largest
    cluster_boxes_data.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
    for box in cluster_boxes_data:
        width = box[2] - box[0]
        height = box[3] - box[1]
        added = False
        for cluster in clusters:
            if cluster.can_add_task(box, width, height):
                cluster.add_task(box, width, height, box[4])
                added = True
                break
        if not added:
            new_cluster = Cluster(box, width, height, box[4])
            clusters.append(new_cluster)
    
    #batch tasks
    priority = 0
    clusters.sort(key=lambda x : x.min_dist)
    for cluster in clusters:
        #make all tasks images same dimensions 
        cluster.make_same_size()
        task_list = cluster.get_tasks()
        batch_list = []
        #make the batch list
        for task in task_list:
            next_task = TaskEntity(image_path=frame.path, coord=task[:4], depth = task[4])
            batch_list.append(next_task)
        #add to return list
        task_batch = TaskBatch(batch_list, batch_list[0].img_width, batch_list[0].img_height, priority = priority)
        task_batches.append(task_batch)
        priority += 1
    
    return task_batches

    
