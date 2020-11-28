from scheduling.misc import *
from scheduling.TaskEntity import *


# read the input cluster box data from file
box_info = read_json_file('../dataset/depth_clustering_detection_flat.json')


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
    cluster_boxes_data.sort(key=lambda x : (x[2] - x[0]) * (x[3] - x[1])) # order based on smaller area first
    #task_list = []
    priority = 0
    for box in cluster_boxes_data:
        task = TaskEntity(image_path=frame.path, coord=box[:4], depth=box[4])
        task_batch = TaskBatch([task], task.img_width, task.img_height, priority = priority)
        task_batches.append(task_batch)
        priority += 1

    return task_batches

# Below is the output after running the code above
#Scheduling history saved.
#deadline miss rate is:  0.033417402269861285
#[52, 65, 36, 38, 24, 27, 21, 25, 0, 0]
#average coverage: 0.712
#average accuracy: 0.545
