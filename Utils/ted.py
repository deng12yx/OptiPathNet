from zss import simple_distance, Node


def build_tree(root_label, edge_dict):
    # 创建根节点
    root = Node(root_label)
    # 创建节点字典，用于快速查找节点对象
    nodes = {root_label: root}
    if root_label not in edge_dict:
        return root
    for node in edge_dict[root_label]:
        # 递归创建子树
        root.addkid(build_tree(node, edge_dict))
    return root
if __name__ == '__main__':

    # 示例数据：根节点和边列表
    root_label = 'f'
    edges = [('f', 'a'), ('a', 'h'), ('a', 'c'), ('h', 'l'), ('f', 'e'),('e', 'b'), ('b', 'd'), ('b', 'g')]
    edge_dict = {}
    for edge in edges:
        if edge[0] not in edge_dict:
            edge_dict[edge[0]] = []
        edge_dict[edge[0]].append(edge[1])
    # 构建树结构
    tree = build_tree(root_label, edge_dict)

    # 打印树结构
    print(f"tree 的结构是{tree}")

    # 示例数据B
    root_label_B = 'f'
    edges_B = [('f', 'a'), ('a', 'd'), ('a', 'c'), ('c', 'b'), ('f', 'e')]
    edge_dict_B = {}
    for edge in edges_B:
        if edge[0] not in edge_dict_B:
            edge_dict_B[edge[0]] = []
        edge_dict_B[edge[0]].append(edge[1])
    tree_B = build_tree(root_label_B, edge_dict_B)

    # 打印树结构
    print(f"treeB 的结构是{tree_B}")

    # 计算简单距离
    distance = simple_distance(tree, tree_B)
    print("Simple Distance:", distance)
