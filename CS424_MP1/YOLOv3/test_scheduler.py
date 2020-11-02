import time
from scheduling.Scheduler import *
import matplotlib.pyplot as plt

start = time.time()

scheduler = Scheduler()
scheduler.run()
scheduler.print_history()
scheduler.visualize_history()

end = time.time()

print("Elapsed time: %f s" % (end - start))


# # example for using visualize_history_file()
history = read_json_file("scheduling_history.json")
# visualize_history_file(history)
# # calculate group average response time from history file
group_response_time = get_group_avg_response_time(history)
x_axis = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
plt.bar(x_axis, group_response_time)
plt.xlabel('Distance')
plt.ylabel('Average Response Time')
plt.title('Average Response Time of bbox')
plt.savefig('group-response-time-plot')
