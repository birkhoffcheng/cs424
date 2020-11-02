from scheduling.misc import *
from scheduling.TaskEntity import *

# read the input bounding box data from file
box_info = read_json_file('../dataset/waymo_ground_truth_flat.json')

box_dict = {}
frame_cnt = 0

def process_frame(frame):
    cluster_boxes_data = get_bbox_info(frame, box_info)
    cluster_boxes_data.sort(key=lambda x : x[4])
    task_list = []
    priority = 0
    for box in cluster_boxes_data:
        if should_skip(box):
            continue
        else:
            task = TaskEntity(image_path=frame.path, coord=box[:4], priority=priority, depth=box[4])
            task_list.append(task)
            priority += 1
    global frame_cnt
    frame_cnt += 1
    return task_list

def should_skip(box):
    global frame_cnt
    #check all recently processed frames
    for pair in box_dict.keys():
        diffx = abs(box[0] - pair[0])
        diffy = abs(box[1] - pair[1])
        #if difference in pixels is below a threshold then consider it the same object
        if diffx + diffy < 5:
            if frame_cnt - box_dict[pair] < 10:
                #skip if processed less than 10 frames ago
                return True
            else:
                #if processed more than 10 frames ago replace (x,y) coordinate key and insert current frame
                box_dict.pop(pair)
                box_dict[(box[0], box[1])] = frame_cnt
                return False
    #if no similar box was found add to dictionary
    box_dict[(box[0], box[1])] = frame_cnt
    return False