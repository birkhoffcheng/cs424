from scheduling.misc import *
from scheduling.TaskEntity import *

# read the input bounding box data from file
box_info = read_json_file('../dataset/waymo_ground_truth_flat.json')

def process_frame(frame):
    """Process frame for scheduling.

    Process a image frame to obtain cluster boxes and corresponding scheduling parameters
    for scheduling.

    Student's code here.

    Args:
        param1: The image frame to be processed.

    Returns:
        A list of tasks with each task containing image_path and other necessary information.
    """

    cluster_boxes_data = get_bbox_info(frame, box_info)
    cluster_boxes_data.sort(key=lambda x : (x[2] - x[0]) * (x[3] - x[1]))
    task_list = []
    priority = 0
    for box in cluster_boxes_data:
        task = TaskEntity(image_path=frame.path, coord=box[:4], priority=priority, depth=box[4])
        task_list.append(task)
        priority += 1
    return task_list
