from scheduling.misc import *
from scheduling.TaskEntity import *

# read the input bounding box data from file
box_info = read_json_file('../dataset/waymo_ground_truth_flat.json')

def process_frame(frame):
    cluster_boxes_data = get_bbox_info(frame, box_info)
    cluster_boxes_data.sort(key=lambda x : x[4])
    task_list = []
    priority = 0
    for box in cluster_boxes_data:
        task = TaskEntity(image_path=frame.path, coord=box[:4], priority=priority, depth=box[4])
        task_list.append(task)
        priority += 1
    return task_list
