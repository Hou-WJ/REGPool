import os


def set_gpu(gpus):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # 通过设置它来限制程序所能看到的可用 GPU 设备列表，从而确保程序只使用指定的 GPU 设备。
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def max_node_nums(data):
    max_nums = 0
    for graph in data:
        if graph.num_nodes >= max_nums:
            max_nums = graph.num_nodes
    return max_nums
