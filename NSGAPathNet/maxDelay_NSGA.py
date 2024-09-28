import copy
import json
import multiprocessing
import random
from multiprocessing.pool import ThreadPool

import networkx as nx
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.problems import get_problem
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import time
from pymoo.core.problem import Problem

import pymoo.gradient.toolbox as anp
import numpy as np
from zss import simple_distance, Node

from ted import build_tree
from topoInfer import path_length, RNJ, generateTree


class DelayProblem(Problem):
    def __init__(self, delayPredict, delayIfAdd, delayNewArray, delayPreArray, shortest_path, leaf, another_node,
                 arraylength):
        super().__init__(
            n_obj=3,  # 目标函数的数量
            n_ieq_constr=2,  # g 约束条件数量
            elementwise_evaluation=True  # 是否逐个评估目标函数值
        )
        self.delayPredict = delayPredict
        self.delayIfAdd = delayIfAdd
        self.delayPreArray = copy.deepcopy(delayPreArray[:arraylength])
        self.delayNewArray = copy.deepcopy(delayNewArray[:arraylength])
        self.shortest_path = shortest_path
        self.pool = multiprocessing.Pool()
        self.leaf = leaf
        self.another_node = another_node
        self.update_keys = [key for key, should_add in delayIfAdd.items() if should_add]
        self.no_update_keys = [key for key, should_add in delayIfAdd.items() if not should_add]

    def _evaluate(self, x, out, *args, **kwargs):
        """

        :param x:
        :param out:
        :param args: [0]:delayPredict, [1]:delayIfAdd
        :param kwargs:
        :return:
        """
        x = x.T
        # print(f"res.X[0] is {res.X[0]}")
        update_values = [self.delayPreArray[-1][key] + x[count] for count, key in enumerate(self.update_keys)]
        no_update_values = [self.delayPreArray[-1][key] + np.zeros((x.shape[1])) for key in self.no_update_keys]
        updated_values = update_values + no_update_values
        self.delayNewArray[-1] = {key: value for key, value in
                                  zip(self.update_keys + self.no_update_keys, updated_values)}
        args = (self.delayNewArray, self.delayPreArray, self.shortest_path, self.leaf, self.another_node)
        # 创建进程池
        results = [self.pool.apply_async(process_task, (i,) + args) for i in range(x.shape[1])]
        all_results = [result.get() for result in results]
        f1, f2, f3 = zip(*all_results)
        g1 = [item - 1 for item in f1]
        g2 = [item - 5 for item in f2]
        out["F"] = anp.column_stack([f1, f2, f3])
        out["G"] = anp.column_stack([g1, g2])


def process_task(i, delayNewArray, delayPreArray, shortest_path, leaf, another_node):
    # delayAddSingle = {key: delayAdddict[key][i] for key in delayAdddict.keys()}

    delayNew = copy.deepcopy(delayNewArray)
    delayPre = copy.deepcopy(delayPreArray)

    if i != -1:
        for key in delayNewArray[-1].keys():
            delayNew[-1][key] = delayNewArray[-1][key][i]
    delaydict = [delayPre, delayNew]
    tree = []
    S = [min(leaf)]
    preD = [node for node in leaf if node != S[0]]
    T = [np.zeros((max(preD) + 1, len(delaydict[j]))) for j in range(len(delaydict))]

    all_G = []
    rho_ij = []
    for j in range(len(delaydict)):
        D = preD.copy()
        for i in range(len(delaydict[j])):
            for node in D:
                T[j][node][i] = path_length(delaydict[j][i], node, shortest_path)
        rnj_topo = RNJ(S, D, T[j])
        Topo, d_length = rnj_topo.topoGenerate()
        # print(f"(V,E) is {Topo}, d_length is {d_length}")
        rho_ij.append(rnj_topo.rho_ij_pre)
        edge_dict = {}
        for edge in Topo[1]:
            if edge[0] not in edge_dict:
                edge_dict[edge[0]] = []
            edge_dict[edge[0]].append(edge[1])
        # 构建树结构
        tree.append(build_tree(S[0], edge_dict))
        G = nx.Graph()
        for edge, weight in d_length.items():
            G.add_edge(edge[0], edge[1], weight=weight)
        all_G.append(G)
        # nx.draw(G,with_labels=True)
        # plt.title("demo")
        # plt.show()

    tree_3 = generateTree(delayPreArray[0].keys(), S)
    # 构建树结构
    tree.append(build_tree(S[0], tree_3))
    simi = 1 - simple_distance(tree[0], tree[1]) / (
            simple_distance(tree[0], Node(1)) + simple_distance(tree[1], Node(1)))
    # print(f"相似度为{simi}")

    # simi = 1-simple_distance(tree[0],tree[2])/(simple_distance(tree[0],Node(1))+simple_distance(tree[2],Node(1)))
    # print(f"真实预测相似度为{simi}")
    # simi = 1-simple_distance(tree[2],tree[1])/(simple_distance(tree[2],Node(1))+simple_distance(tree[1],Node(1)))
    # print(f"混淆预测相似度为{simi}")
    cost = 0
    middleNode = another_node
    length = len(delaydict[0])
    for i in range(len(delaydict[0])):
        cost_single = 0
        for node in middleNode:
            delay1 = max(path_length(delaydict[0][i], node, shortest_path), 0.000000001)
            delay2 = max(path_length(delaydict[1][i], node, shortest_path), 0.000000001)
            cost_single += delay2 / delay1
            cost_single -= 1
        if cost_single < 1000:
            cost_single /= len(middleNode)
            cost += cost_single
        else:
            length -= 1

    # print(f"length is {len(delaydict[0])}")
    if length == 0:
        cost = 100
    else:
        cost /= length
    # print(f"相似度为{simi},成本为{cost}")

    max_weight_edge = []
    nx.write_graphml(all_G[0], "img/large_pre_ran.graphml")
    nx.write_graphml(all_G[1], "img/large_lat_ran.graphml")
    for G in all_G:
        edge_weights = nx.get_edge_attributes(G, 'weight')
        max_weight_edge.append(max(edge_weights, key=edge_weights.get))
    composeG = nx.compose(all_G[0], all_G[1])
    pathLen = -(nx.shortest_path_length(composeG, source=max_weight_edge[0][0],
                                        target=max_weight_edge[1][0])
                + nx.shortest_path_length(composeG, source=max_weight_edge[0][1],
                                          target=max_weight_edge[1][1])) / 2
    return simi, cost, pathLen


def read_list_from_txt(filename):
    try:
        # 打开文件
        with open(filename, 'r') as file:
            # 读取文件内容
            content = file.read()
            # 将字符串转换为列表
            my_list = eval(content)
            return my_list
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None


def find_last_common_element(list1, list2):
    # 从前向后遍历两个列表
    i = 1
    j = 1
    while i < len(list1) and j < len(list2):
        # 如果找到了不同的元素，返回前一个元素
        if list1[i] != list2[j]:
            return list1[i - 1]
        i += 1
        j += 1


class NewdelayCompute:
    def __init__(self, weight_info, delay_info):
        self.leaf = None
        self.another_node = []
        self.weight_info = weight_info
        self.delay_info = delay_info
        self.delayPreArray = read_list_from_txt("./1/Deltacom/delayArray_Deltacom_9.0.txt")
        self.delayNewArray = read_list_from_txt("./1/Deltacom/newDelayArray_Deltacom_9.0.txt")
        self.arrayLen = [140]
        self.allF = []
        self.allTime = []
        self.shortest_path = {}
        self.allX = []
        self.popsize = [10]

    def generate_max_weight_edges(self):
        weight_neighbor = {}
        for links, weight in self.weight_info.items():
            src, dst = links
            if src not in weight_neighbor.keys():
                weight_neighbor[src] = {}
            if dst not in weight_neighbor.keys():
                weight_neighbor[dst] = {}
            weight_neighbor[src][dst] = weight
            weight_neighbor[dst][src] = weight
        # print(f"weight is {weight_neighbor}")
        max_weight = {}
        for key in weight_neighbor.keys():
            max_weight[key] = {}
            for next_key in weight_neighbor[key].keys():
                max_weight[key][next_key] = [weight_neighbor[key][next_key], (key, next_key)]
                # 定义一个队列
                queue = [next_key]
                # 将节点加入队列
                # 定义一个遍历过的节点集合
                visited = set()
                visited.add(key)
                visited.add(next_key)
                while queue:
                    node = queue.pop(0)
                    visited.add(node)
                    for next_node in weight_neighbor[node].keys():
                        if next_node not in visited:
                            if weight_neighbor[node][next_node] > max_weight[key][next_key][0]:
                                max_weight[key][next_key] = [weight_neighbor[node][next_node], (node, next_node)]
                            queue.append(next_node)
        # print("return ")
        return max_weight

    def newDelayCompute(self):
        """
        :param max_weight:
        :param delay_info:
        :return:
        """
        max_weight = self.generate_max_weight_edges()
        delayPredict = {}
        delayIfAdd = {}
        # delayNeedAdd = {}
        maxDelay = 0
        for node1 in max_weight.keys():
            for node2 in max_weight[node1].keys():
                delayPredict[(node1, node2)] = self.delay_info[node1][node2]
                delayIfAdd[(node1, node2)] = False
                maxDelay = max(maxDelay, self.delay_info[node1][node2])
        self.delayPreArray.append(delayPredict)
        self.delayPreArray.pop(0)
        new_delay = {}
        new_node = max(list(max_weight.keys()))
        # print(f"max_weight is {max_weight}")
        # print(f"delay_info is {delay_info}")
        for key in max_weight.keys():
            if key not in new_delay.keys():
                new_delay[key] = {}
            max_delay_key = max([max_weight[key][item][0] for item in max_weight[key].keys()])
            max_delay_link = [max_weight[key][item][1] for item in max_weight[key].keys()
                              if max_weight[key][item][0] == max_delay_key][0]
            for item in max_weight[key].keys():
                if max_weight[key][item][0] != max_delay_key:
                    # print("1")
                    if item not in new_delay.keys():
                        new_delay[item] = {}
                    node1 = max_delay_link[0]
                    node2 = max_delay_link[1]
                    # new_delay[key][item] = (delay_info[key][item] + delay_info[node1][node2]) / 2
                    new_delay[key][item] = random.uniform(self.delay_info[key][item], maxDelay)
                    # new_delay[item][key] = new_delay[key][item]
                    delayIfAdd[(key, item)] = True
                elif key in max_delay_link and item in max_delay_link:
                    # print("2")
                    new_node += 1
                    new_delay[key][item] = self.delay_info[key][item]
                    new_delay[key][new_node] = self.delay_info[key][item]
                    new_delay[new_node] = {}
                    new_delay[new_node][item] = self.delay_info[key][item]
                    new_delay[new_node][key] = self.delay_info[key][item]
                    if item not in new_delay.keys():
                        new_delay[item] = {}
                    new_delay[item][new_node] = self.delay_info[key][item]
                else:
                    # print("4")
                    new_delay[key][item] = self.delay_info[key][item]
        G = nx.Graph()
        G.add_edges_from(delayPredict.keys())
        diameter = nx.diameter(G)

        print("Diameter:", diameter)
        # nx.draw(G, with_labels=True)
        # plt.show()
        self.leaf = [node for node in G.nodes() if G.degree(node) == 1]
        # self.another_node = [node for node in G.nodes() if G.degree(node) != 1]
        source = min(self.leaf)
        dest = [target for target in self.leaf if target != source]
        for i in range(len(dest)):
            for j in range(i + 1, len(dest)):
                # shortest_paths = [nx.shortest_path(G, source, target_node) for target_node in dest]
                last_common_node = find_last_common_element(nx.shortest_path(G, source, dest[i]),
                                                            nx.shortest_path(G, source, dest[j]))
                self.another_node.append(last_common_node)
        print(f"leaf is {self.leaf}")
        print(f"another_node is {self.another_node}")
        # self.shortest_path = {}
        for target in [target for target in G.nodes() if target != source]:
            self.shortest_path[str(target)] = nx.shortest_path(G, source=source, target=target)
        # f1, f2, f3 = process_task(-1, self.delayNewArray[100:], self.delayPreArray[100:], self.shortest_path, self.leaf,
        #                           self.another_node)
        # print(f"f1 is {f1}")
        self.delayNewArray = self.delayNewArray[100:]
        self.delayPreArray = self.delayPreArray[100:]
        for eva in [50]:
            for pop_size in [20]:
                # if startIndex<len(self.delayNewArray):
                delayProblem = DelayProblem(delayPredict, delayIfAdd, self.delayNewArray, self.delayPreArray,
                                            self.shortest_path, self.leaf, self.another_node, 40)
                delayProblem.n_var = sum(value for value in delayIfAdd.values())
                delayProblem.xl = np.array([0] * delayProblem.n_var)
                # delayProblem.n_ieq_constr = delayProblem.n_var
                delayProblem.xu = np.array([maxDelay] * delayProblem.n_var)
                # print(f"xu is {delayProblem.xu}")
                # pool = ThreadPool(8)
                algorithm = NSGA2(pop_size=pop_size,  # 种群数量
                                  n_offsprings=pop_size,  # 后代数量
                                  eliminate_duplicates=True  # 确保后代的目标是不同
                                  )
                start = time.time()
                res = minimize(delayProblem,
                               algorithm,
                               ('n_gen', eva),
                               verbose=False)
                end = time.time()
                # print(f"n is {pop_size}, population is {res.F[0]}, time is {end - start}")
                # with(open("result_big_nsga.json", "a")) as f:
                #     json.dump({"n": pop_size, "eva": eva, "score": list(res.F[0]), "time": end - start}, f)
                #     f.write("\n")
                # self.allTime.append(end - start)
                # self.allF.append(res.F)
                #
                # print('耗时：', end - start, '秒')
                # print(f"Best solution found:popsize is {10} \nF = {res.F}")
                # self.allX.append(res.X[0])
                # # res.X = res.X.T
                # update_keys = [key for key, should_add in delayIfAdd.items() if should_add]
                # no_update_keys = [key for key, should_add in delayIfAdd.items() if not should_add]
                # for i in range(40, 1000, 40):
                #     if i + 40 < len(self.delayPreArray):
                #         delay_dict = self.delayPreArray[i + 40]
                #         # print(f"res.X[0] is {res.X[0]}")
                #         update_values = [delay_dict[key] + res.X[0][count] for count, key in enumerate(update_keys)]
                #         no_update_values = [delay_dict[key] for key in no_update_keys]
                #         updated_values = update_values + no_update_values
                #         self.delayNewArray[i + 40] = {key: value for key, value in
                #                                       zip(update_keys + no_update_keys, updated_values)}
                # with(open("result_middle_nsga_delay.json", "a")) as f:
                #     f.write(self.delayNewArray)
                #     f.write("\n")
        # for packetsLength in range(1000, 10000, 1000):
        #     simi, cost, pathLen = process_task(-1, self.delayNewArray[:packetsLength],
        #                                            self.delayPreArray[:packetsLength], self.shortest_path, self.leaf,
        #                                            self.another_node)
        #     print(f"simi is {simi},cost is {cost},pathLen is {pathLen}")

        return new_delay


def generate_flow_rules(G, sorted_links_scores, max_link_delays, score_threshold, node_to_port, λ, μ):
    """
    根据网络拓扑结构、链路分数、链路延迟和阈值生成流规则。

    :param G: 网络拓扑图。
    :param sorted_links_scores: 按分数排序的链路字典，包括分数。
    :param max_link_delays: 从每个端口经过的最大链路延迟字典。
    :param score_threshold: 链路分数阈值（高于该值的链路被视为重要）。
    :param node_to_port: 节点到输出端口的映射。
    :param λ: 计算流规则的参数（1,2）。
    :param μ: 计算流规则的参数。
    :return: flow_rules（生成的流规则字典）
    """
    flow_rules = {}  # 存储生成的流规则
    # 遍历网络中的每个节点
    for node in G.nodes():
        flow_rules[node] = {}  # 初始化该节点的流规则字典
        # 遍历节点的邻居节点
        for node_neighbor in G[node].neighbors():
            out_port = node_to_port[(node, node_neighbor)]  # 起始端口
            in_port = node_to_port[(node_neighbor, node)]  # 终止端口
            # 检查链路分数是否高于阈值，如果高于阈值，则认为该链路重要
            if sorted_links_scores[(out_port, in_port)] >= score_threshold:
                # 根据参数λ计算要增加的延迟，这里生成两个
                flow_rules[node][out_port] = 1
            # 如果链路不重要但该出口会经过重要链路
            elif max_link_delays[out_port]['delay'] >= score_threshold:
                flow_rules[node][out_port] = 2
                # 如果以上条件均不满足，则使用链路默认延迟作为流规则
            else:
                flow_rules[node][out_port] = 0  # 默认情况
    return flow_rules


def calculate_growth_rate(old_value, new_value):
    growth_rate = ((new_value - old_value) / old_value) * 100
    return growth_rate


def calculate_decline_rate(old_value, new_value):
    decline_rate = ((old_value - new_value) / old_value) * 100
    return decline_rate


if __name__ == '__main__':
    weight_info = read_list_from_txt("./1/Deltacom/weight_info_Deltacom_1.0.txt")
    delay_info = read_list_from_txt("./1/Deltacom/delayinfo_Deltacom_1.0.txt")
    newDelay = NewdelayCompute(weight_info, delay_info)
    new_delay = newDelay.newDelayCompute()
