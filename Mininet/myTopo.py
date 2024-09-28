import networkx as nx
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import RemoteController

import networkx as nx

class MyTopo(Topo):
    def __init__(self):
        Topo.__init__(self)
        file_path = "./sources/Arn.graphml"
        G = nx.read_graphml(file_path)
        G=nx.Graph(G)
        hash_nodes = {}  # 用于存储节点和主机
        hosts = set()
        switches = set()

        # 遍历图中的所有节点
        for i, node in enumerate(G.nodes):
            degree = G.degree[node]  # 获取节点的度数
            if degree == 1:
                hosts.add(i + 1)  # 将度数为1的节点添加到主机集合中
                hash_nodes[node] =f'h{i+1}'
            else:
                switches.add(i + 1)  # 将度数大于1的节点添加到交换机集合中
                hash_nodes[node] =f's{i+1}'

        # 添加交换机节点
        for switch in switches:
            self.addSwitch(f's{switch}')

        # 添加主机节点并连接到交换机
        for host in hosts:
            self.addHost(f'h{host}')

        # 添加连接
        for edge in G.edges:
            node1, node2 = edge
            self.addLink(f'{hash_nodes[node1]}', f'{hash_nodes[node2]}')



topos={'mytopo':(lambda : MyTopo())}