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
    # student's code here.
    # determine priority based on area size of bounding boxes
    cluster_boxes_index = 0
    area_list = []
    # create list of frame's bounding boxes with [area, index]
    for bounding_box_data in cluster_boxes_data:
        # compute area x*y
        area = (bounding_box_data[2] - bounding_box_data[0]) * (bounding_box_data[3] - bounding_box_data[1])
        area_list.append([area,cluster_boxes_index])
        cluster_boxes_index = cluster_boxes_index + 1
    # sort bounding box list according to area priority, (small area to large area)
    area_list.sort(key=lambda x : x[0])
    task_list = []
    priority_count = 0
    # create task list based on area priority
    for bounding_box in area_list:
        task = TaskEntity(image_path = frame.path, coord = cluster_boxes_data[bounding_box[1]][:4], priority = priority_count, depth = cluster_boxes_data[bounding_box[1]][4])
        task_list.append(task)
        priority_count = priority_count + 1

    return task_list
