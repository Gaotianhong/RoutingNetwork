import sys
import random
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from netaddr import IPNetwork
from PyQt5 import QtWidgets
import mainwindow


class Network(mainwindow.Ui_MainWindow):
    def __init__(self, MainWindow):
        super().setupUi(MainWindow)
        self.n = np.random.randint(150, 201)  # 路由器数量
        self.n = 50
        self.p = 0.05  # 生成边的概率
        self.G = nx.Graph()

        self.create_graph.clicked.connect(self.create_choose_graph)  # 生成网络
        self.describe_graph.clicked.connect(self.describe_network)  # 统计描述网络
        self.top_router.clicked.connect(self.generate_top10_routing_table)  # Top10路由器

        self.addV.clicked.connect(self.add_vertex)  # 添加顶点
        self.delV.clicked.connect(self.del_vertex)  # 删除顶点
        self.addE.clicked.connect(self.add_edge)  # 添加边
        self.delE.clicked.connect(self.del_edge)  # 删除边
        self.change_weight.clicked.connect(self.reassign_weight)  # 对指定边重新赋以新的权重
        self.routing_table.clicked.connect(self.generate_routing_table)  # 生成路由表

        self.draw_network.clicked.connect(self.draw)  # 绘制网络
        self.draw_degree.clicked.connect(self.draw_degree_scatter)  # 绘制节点度分布
        self.draw_min_tree.clicked.connect(self.draw_minimum_spanning_tree)  # 绘制最小生成树

    @staticmethod
    def generate_ip_address():
        """生成同一个AS的IP地址"""
        AS_IP = []
        for ip in IPNetwork('192.0.2.0/24'):
            AS_IP.append(str(ip))
        random.shuffle(AS_IP)
        AS_IP = ' '.join(AS_IP).split(' ')
        return AS_IP

    def create_connected_graph(self):
        """创建连通图，n为图的节点个数，p为生成某条边的概率"""
        edges = itertools.combinations(range(self.n), 2)
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))  # 加入节点
        if self.p <= 0:
            return G

        for e in edges:
            if np.random.random() < self.p:
                # G.add_edge(*e, weight=np.random.randint(1, 10))
                self.G.add_edge(*e, weight=np.random.randint(1, 10))

        # 保证该图是连通图
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                # 判断从节点i到j是否有路径，若没有路径则添加一条边保证该图是连通图
                if not nx.has_path(self.G, i, j):
                    # print("has no path")
                    self.G.add_edge(i, j, weight=np.random.randint(1, 10))  # 加入一条路径

        AS_IP = self.generate_ip_address()
        # 分配ip地址
        for i in range(self.n):
            self.G.nodes[i]['IP'] = AS_IP[i]

    def create_internet_graph(self):
        """创建网络自治域连通图"""
        self.G = nx.random_internet_as_graph(self.n)  # 创建网络自治域连通图
        n = nx.number_of_nodes(self.G)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if self.G.has_edge(i, j):
                    # 两个节点之间有边，则赋予权重生成带权图
                    self.G.add_edge(i, j, weight=np.random.randint(1, 10))

        AS_IP = self.generate_ip_address()
        # 分配ip地址
        for i in range(n):
            self.G.nodes[i]['IP'] = AS_IP[i]

    def create_choose_graph(self):
        choice, ok = QtWidgets.QInputDialog.getText(None, "生成网络",
                                                    "请选择生成网络的策略：\n【1】以添加边的形式创建网络\n【2】创建网络自治域连通图")
        if not ok:
            return
        if int(choice) == 1:
            self.create_connected_graph()
        elif int(choice) == 2:
            self.create_internet_graph()
        else:
            QtWidgets.QMessageBox.warning(None, "提示", "\n输入内容有误😊！")

    def describe_network(self):
        """对网络进行统计描述"""
        G = self.G
        if nx.is_empty(G):
            QtWidgets.QMessageBox.information(None, "提示", "\n您没有生成网络，不能进行此操作😊！")
            return
        self.textEdit.clear()
        self.textEdit.append("\t\t************对网络进行统计描述************")
        self.textEdit.append("节点数：{}\t边数：{}".format(G.number_of_nodes(), G.number_of_edges()))
        # self.textEdit.append("图的顶点集：{}".format(G.nodes()))
        # self.textEdit.append("图的边集：{}".format(G.edges()))
        # self.textEdit.append("图的顶点属性：{}".format(G.nodes.data()))
        # self.textEdit.append("图的邻接表：{}".format(list(G.adjacency())))
        self.textEdit.append("图的邻接矩阵：\n{}".format(np.array(nx.adjacency_matrix(G).todense())))
        # self.textEdit.append("列表字典为：{}".format(nx.to_dict_of_lists(G)))
        T = nx.minimum_spanning_tree(G)  # 最小生成树
        w = nx.get_edge_attributes(T, 'weight')  # 提取字典数据
        self.textEdit.append("最小生成树为：{}".format(w))
        self.textEdit.append("最小生成树的长度为：{}".format(sum(w.values())))
        self.textEdit.append("节点的度：{}".format(nx.degree(G)))
        degree_sum = sum(nx.degree_histogram(G))
        degree_dict = {}
        self.textEdit.append("节点的度的分布情况：")
        for i in range(len(nx.degree_histogram(G))):
            if nx.degree_histogram(G)[i] != 0:  # nx.degree_histogram(G)[i] 表示度为i的节点个数
                degree_dict[i] = nx.degree_histogram(G)[i] / degree_sum
        self.textEdit.append("{}".format(degree_dict))
        self.textEdit.append(
            "网络的平均度：{}".format(round(sum(np.array(nx.degree(G))[:, 1]) / G.number_of_nodes()), 3))  # 网络的平均度
        self.textEdit.append("网络直径：{}".format(nx.diameter(G)))
        self.textEdit.append("平均路径长度：{}".format(round(nx.average_shortest_path_length(G), 3)))
        Ci = nx.clustering(G)  # 节点vi的ki个邻居节点之间实际存在的边数和总的可能C(ki,2)的边数之比
        cluster = {}
        for index, value in enumerate(Ci.values()):
            cluster[index] = round(value, 3)
        self.textEdit.append("各个顶点的聚类系数：{}".format(cluster))
        self.textEdit.append("整个网络的聚类系数：{}".format(round(nx.average_clustering(G), 3)))

    def floyd(self):
        """Floyd算法"""
        graph = np.array(nx.adjacency_matrix(self.G).todense()).astype(float)  # 获取图的邻接矩阵
        n = nx.number_of_nodes(self.G)  # 节点个数

        for i in range(n):
            for j in range(n):
                if i != j and graph[i][j] == 0:
                    graph[i][j] = np.inf  # 节点之间没有边
        dis = graph
        path = np.zeros((n, n))  # 路由矩阵初始化
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dis[i][k] + dis[k][j] < dis[i][j]:
                        dis[i][j] = dis[i][k] + dis[k][j]
                        path[i][j] = k
        return dis, path.astype(int)

    def get_closeness_centrality(self):
        """近性中心度：v到其他各个节点的最短路径的长度之和的倒数"""
        G = self.G
        # print("Call library:", nx.closeness_centrality(G, distance='weight'))
        return nx.closeness_centrality(G, distance='weight')
        closeness_centrality = {}
        n = nx.number_of_nodes(G)  # 节点个数
        dis, path = self.floyd()  # floyd算法求所有顶点对之间对最短距离
        for i in range(n):
            closeness_centrality[i] = (n - 1) / sum(dis[i, :])
        return closeness_centrality

    def get_betweenness_centrality(self):
        """图中任意两个节点对之间的最短路径当中，其中经过v的最短路径所占比例"""
        G = self.G
        # print("Call library:", nx.betweenness_centrality(G, weight='weight'))
        return nx.betweenness_centrality(G, weight='weight')
        betweenness_centrality = {}
        n = nx.number_of_nodes(G)  # 节点个数
        for i in range(n):
            betweenness_centrality[i] = 0  # 初始化字典
        for i in range(n):
            for j in range(n):
                try:
                    dp = list(nx.all_shortest_paths(G, i, j, weight='weight'))  # 获得两个节点之间所有最短路径
                except Exception:
                    pass
                if len(dp) > 1:  # 有多条最短路径
                    for m in range(len(dp)):
                        for k in range(1, len(dp[m]) - 1):
                            betweenness_centrality[dp[m][k]] = betweenness_centrality[dp[m][k]] + 1 / len(dp)
                else:
                    if len(dp[0]) > 2:  # 两节点之间的最短路径不是直接相连
                        for k in range(1, len(dp[0]) - 1):
                            betweenness_centrality[dp[0][k]] = betweenness_centrality[dp[0][k]] + 1
        for i in range(n):
            betweenness_centrality[i] = betweenness_centrality[i] / ((n - 1) * (n - 2))  # 归一化
        return betweenness_centrality

    def evaluate_importance(self):
        """评价网络中路由器的重要性（结合近性中心性和介性中心性）"""
        G = self.G
        if nx.is_empty(G):
            QtWidgets.QMessageBox.information(None, "提示", "\n您没有生成网络，不能进行此操作😊！")
            return
        closeness_centrality = self.get_closeness_centrality()
        betweenness_centrality = self.get_betweenness_centrality()
        # print("closeness centrality:", closeness_centrality)
        # print("betweenness centrality:", betweenness_centrality)
        n = nx.number_of_nodes(G)
        importance_rank = {}
        for i in range(n):
            if G.has_node(i):
                importance_rank[i] = closeness_centrality[i] * betweenness_centrality[i]
        return importance_rank

    def generate_routing_table(self):
        """生成路由表"""
        G = self.G
        if nx.is_empty(G):
            QtWidgets.QMessageBox.information(None, "提示", "\n您没有生成网络，不能进行此操作😊！")
            return
        dp = nx.shortest_path(G, weight='weight')  # 获得网络中两个节点之间所有最短路径
        # print(dp)
        self.textEdit.clear()
        for k, v in dp.items():  # k代表路由器编号
            self.textEdit.append("routing table for N{}, IP address:{}".format(k, G.nodes[k]['IP']))
            # print(v)
            self.textEdit.append("目的网络\t距离\t下一跳路由器")
            for kk, vv in v.items():
                if k != kk:
                    dist = nx.shortest_path_length(G, k, kk, weight='weight')
                    self.textEdit.append("  N{}\t {}\t   N{}".format(kk, dist, vv[1]))

    def generate_top10_routing_table(self):
        """输出重要性Top10的路由器的路由表"""
        G = self.G
        if nx.is_empty(G):
            QtWidgets.QMessageBox.information(None, "提示", "\n您没有生成网络，不能进行此操作😊！")
            return
        num = 10
        if nx.number_of_nodes(G) < num:
            print("节点数量小于10")
            return
        self.textEdit.clear()
        importance = self.evaluate_importance()  # 评价节点重要性
        # print(importance)
        importance_rank = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        # print(importance_rank)
        TopRouters = []
        for i in range(num):
            TopRouters.append(list(importance_rank[i])[0])  # 路由器编号
        for k in TopRouters:
            dp = nx.shortest_path(G, source=k, weight='weight')
            # print(dp)
            self.textEdit.append("routing table for N{}, IP address:{}".format(k, G.nodes[k]['IP']))
            self.textEdit.append("目的网络\t距离\t下一跳路由器")
            for kk, vv in dp.items():
                if k != kk:
                    dist = nx.shortest_path_length(G, k, kk, weight='weight')
                    self.textEdit.append("  N{}\t {}\t   N{}".format(kk, dist, vv[1]))

        self.textEdit.append("Top10路由器编号是：{}".format(TopRouters))

    def add_vertex(self):
        G = self.G
        if nx.is_empty(G):
            QtWidgets.QMessageBox.information(None, "提示", "\n您没有生成网络，不能进行此操作😊！")
            return
        vertex, ok = QtWidgets.QInputDialog.getText(None, "添加顶点", "请输入需要增加的顶点：")
        if not ok:
            return
        if G.has_node(int(vertex)):
            QtWidgets.QMessageBox.information(None, "提示", "\n网络中已存在该节点，添加失败😊！")
            return
        G.add_node(int(vertex))

    def del_vertex(self):
        G = self.G
        if nx.is_empty(G):
            QtWidgets.QMessageBox.information(None, "提示", "\n您没有生成网络，不能进行此操作😊！")
            return
        node_del, ok = QtWidgets.QInputDialog.getInt(None, "删除顶点", "请输入需要删除的顶点编号：")
        if not ok:
            return
        if G.has_node(node_del):
            G.remove_node(node_del)
        else:
            QtWidgets.QMessageBox.warning(None, "提示", "\n图中不包含该节点，删除失败😊！")

    def add_edge(self):
        G = self.G
        if nx.is_empty(G):
            QtWidgets.QMessageBox.information(None, "提示", "\n您没有生成网络，不能进行此操作😊！")
            return
        edge, ok = QtWidgets.QInputDialog.getText(None, "添加边", "请输入需要增加的边以及对应的权重：")
        if not ok:
            return
        if len(edge.split(' ')) != 3:
            QtWidgets.QMessageBox.information(None, "提示", "输入格式有误😊！")
            return
        u = int(edge.split(' ')[0])
        v = int(edge.split(' ')[1])
        cost = int(edge.split(' ')[2])
        if G.has_edge(u, v):
            QtWidgets.QMessageBox.warning(None, "提示", "\n原边已经存在，请重新选择😊！")
            return
        G.add_edge(u, v, weight=cost)

    def del_edge(self):
        G = self.G
        if nx.is_empty(G):
            QtWidgets.QMessageBox.information(None, "提示", "\n您没有生成网络，不能进行此操作😊！")
            return
        edge, ok = QtWidgets.QInputDialog.getText(None, "删除边", "请输入需要删除的边：")
        if not ok:
            return
        if len(edge.split(' ')) != 2:
            QtWidgets.QMessageBox.information(None, "提示", "输入格式有误😊！")
            return
        u = int(edge.split(' ')[0])
        v = int(edge.split(' ')[1])
        if G.has_edge(u, v):
            G.remove_edge(u, v)
        else:
            QtWidgets.QMessageBox.warning(None, "提示", "\n该边不存在，无法删除😊！")

    def reassign_weight(self):
        G = self.G
        if nx.is_empty(G):
            QtWidgets.QMessageBox.information(None, "提示", "\n您没有生成网络，不能进行此操作😊！")
            return
        edge, ok = QtWidgets.QInputDialog.getText(None, "重新赋以权重", "请输入需要更新的边以及对应的权重：")
        if not ok:
            return
        if len(edge.split(' ')) != 3:
            QtWidgets.QMessageBox.information(None, "提示", "输入格式有误😊！")
            return
        u = int(edge.split(' ')[0])
        v = int(edge.split(' ')[1])
        cost = int(edge.split(' ')[2])
        if G.has_edge(u, v):
            G.add_edge(u, v, weight=cost)
        else:
            QtWidgets.QMessageBox.warning(None, "提示", "\n该边不存在，请重新选择😊！")

    def change_graph(self):
        G = self.G
        if nx.is_empty(G):
            QtWidgets.QMessageBox.information(None, "提示", "\n您没有生成网络，不能进行此操作😊！")
            return
        print("原始图的列表字典为：", nx.to_dict_of_lists(G))
        print("图的邻接表：", list(G.adjacency()))
        while True:
            self.generate_routing_table(G)
            print("请选择你要执行的操作：[1] 顶点操作 [2] 边操作 [3] 退出")
            choice = eval(input("请选择（输入数字）："))
            if choice == 1:
                print("[1]增加顶点 [2]删除顶点 [3]返回")
                node_choice = eval(input("请选择："))
                if node_choice == 1:
                    G.add_node(G.number_of_nodes())
                elif node_choice == 2:
                    node_del = eval(input("请输入需要删除的顶点："))
                    if G.has_node(node_del):
                        G.remove_node(node_del)
                    else:
                        print("图中不包含该节点，删除失败！")
                else:
                    continue
                print("图的列表字典为：", nx.to_dict_of_lists(G))
            elif choice == 2:
                print("[1]增加边 [2]删除边 [3]重新对指定边赋予新的权重 [4]返回")
                edge_choice = eval(input("请选择："))
                if edge_choice == 1:
                    u, v, cost = eval(input("请输入需要增加的边以及对应的权重："))
                    if G.has_edge(u, v):
                        print("原边已经存在，请重新选择！")
                        continue
                    G.add_edge(u, v, weight=cost)
                elif edge_choice == 2:
                    u, v = eval(input("请输入删除的边："))
                    if G.has_edge(u, v):
                        G.remove_edge(u, v)
                    else:
                        print("该边不存在，无法删除！")
                elif edge_choice == 3:
                    u, v, cost = eval(input("请输入需要更新的边以及对应的权重："))
                    if G.has_edge(u, v):
                        G.add_edge(u, v, weight=cost)
                    else:
                        print("该边不存在，请重新选择！")
                else:
                    continue
                print("图的邻接表：", list(G.adjacency()))
            else:
                break
            self.generate_routing_table(G)

    def draw(self):
        """绘制网络"""
        G = self.G
        if nx.is_empty(G):
            QtWidgets.QMessageBox.information(None, "提示", "\n您没有生成网络，不能进行此操作😊！")
            return
        plt.figure(figsize=(10, 7))
        plt.title("网络")
        degree = nx.degree(G)  # 节点的度
        degree_sort = sorted(degree, key=lambda x: x[1], reverse=True)
        color = []
        degree_size = []
        for k in degree:
            if k[1] != 0:
                degree_size.append(k[1] * 10)
            else:
                degree_size.append(30)
            if k[1] >= degree_sort[2][1]:  # 度数排名前3节点画红色
                color.append('red')
            else:
                color.append('cornflowerblue')
        nx.draw_networkx(G, node_size=degree_size, node_color=color, font_size=5, font_weight='bold', width=0.1)
        plt.show()

    def draw_weight_graph(self):
        """绘制带权网络"""
        G = self.G
        if nx.is_empty(G):
            QtWidgets.QMessageBox.information(None, "提示", "\n您没有生成网络，不能进行此操作😊！")
            return
        plt.figure(figsize=(10, 7))
        pos = nx.shell_layout(G)  # 顶点在同心圆上分布
        pos = nx.random_layout(G)  # 顶点随机分布
        # pos = nx.spring_layout(G)  # Fruchterman-Reingold算法排列顶点
        # pos = nx.spectral_layout(G)  # Laplace特征向量排布顶点
        nx.draw_networkx(G, pos, node_size=200, width=0.1)
        w = nx.get_edge_attributes(G, 'weight')
        plt.title("带权网络")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=w)
        plt.show()

    def draw_degree_scatter(self):
        G = self.G
        if nx.is_empty(G):
            QtWidgets.QMessageBox.information(None, "提示", "\n您没有生成网络，不能进行此操作😊！")
            return
        degree_sum = sum(nx.degree_histogram(G))
        degree_dict = {}
        for i in range(len(nx.degree_histogram(G))):
            if nx.degree_histogram(G)[i] != 0:  # nx.degree_histogram(G)[i] 表示度为i的节点个数
                degree_dict[i] = nx.degree_histogram(G)[i] / degree_sum
        plt.figure(dpi=120)
        plt.title("节点度分布情况")
        plt.scatter(degree_dict.keys(), degree_dict.values())
        plt.show()

    def draw_minimum_spanning_tree(self):
        G = self.G
        if nx.is_empty(G):
            QtWidgets.QMessageBox.information(None, "提示", "\n您没有生成网络，不能进行此操作😊！")
            return
        T = nx.minimum_spanning_tree(G)  # 最小生成树
        plt.figure(figsize=(10, 7))
        plt.title("最小生成树")
        nx.draw_networkx(T, node_size=80, font_size=5, font_weight='bold', width=0.1)
        plt.show()

    def k_shell(self, k_importance):
        """k_shell算法"""
        G = self.G
        graph = G.copy()
        ks_dict = {}  # 每个节点的重要性
        ks = 1

        while graph.nodes():
            temp = []  # 暂存度为ks的顶点
            node_degrees_dict = dict(graph.degree())

            while True:
                for k, v in node_degrees_dict.items():
                    if v <= ks:
                        temp.append(k)
                        graph.remove_node(k)  # 删除度数小于等于ks的节点
                node_degrees_dict = dict(graph.degree())
                if ks not in node_degrees_dict.values():
                    break
            ks_dict[ks] = temp
            ks += 1
        for k in list(ks_dict.keys()):  # 寻找指定重要性范围的节点
            if k < k_importance:
                del ks_dict[k]

        return ks_dict


# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    G = Network(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
