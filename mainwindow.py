# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(842, 598)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(40, 80, 120, 100))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.create_graph = QtWidgets.QPushButton(self.layoutWidget)
        self.create_graph.setObjectName("create_graph")
        self.verticalLayout.addWidget(self.create_graph)
        self.top_router = QtWidgets.QPushButton(self.layoutWidget)
        self.top_router.setObjectName("top_router")
        self.verticalLayout.addWidget(self.top_router)
        self.describe_graph = QtWidgets.QPushButton(self.layoutWidget)
        self.describe_graph.setObjectName("describe_graph")
        self.verticalLayout.addWidget(self.describe_graph)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 200, 190, 161))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.addV = QtWidgets.QPushButton(self.layoutWidget1)
        self.addV.setObjectName("addV")
        self.gridLayout_2.addWidget(self.addV, 0, 0, 1, 1)
        self.addE = QtWidgets.QPushButton(self.layoutWidget1)
        self.addE.setObjectName("addE")
        self.gridLayout_2.addWidget(self.addE, 1, 0, 1, 1)
        self.change_weight = QtWidgets.QPushButton(self.layoutWidget1)
        self.change_weight.setObjectName("change_weight")
        self.gridLayout_2.addWidget(self.change_weight, 2, 0, 1, 2)
        self.routing_table = QtWidgets.QPushButton(self.layoutWidget1)
        self.routing_table.setObjectName("routing_table")
        self.gridLayout_2.addWidget(self.routing_table, 3, 0, 1, 2)
        self.delE = QtWidgets.QPushButton(self.layoutWidget1)
        self.delE.setObjectName("delE")
        self.gridLayout_2.addWidget(self.delE, 1, 1, 1, 1)
        self.delV = QtWidgets.QPushButton(self.layoutWidget1)
        self.delV.setObjectName("delV")
        self.gridLayout_2.addWidget(self.delV, 0, 1, 1, 1)
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(200, 10, 631, 581))
        self.textEdit.setObjectName("textEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 370, 171, 51))
        font = QtGui.QFont()
        font.setFamily("Zapfino")
        font.setPointSize(16)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(30, 420, 133, 100))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget2)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.draw_degree = QtWidgets.QPushButton(self.layoutWidget2)
        self.draw_degree.setObjectName("draw_degree")
        self.gridLayout.addWidget(self.draw_degree, 2, 0, 1, 1)
        self.draw_network = QtWidgets.QPushButton(self.layoutWidget2)
        self.draw_network.setObjectName("draw_network")
        self.gridLayout.addWidget(self.draw_network, 1, 0, 1, 1)
        self.draw_min_tree = QtWidgets.QPushButton(self.layoutWidget2)
        self.draw_min_tree.setObjectName("draw_min_tree")
        self.gridLayout.addWidget(self.draw_min_tree, 3, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 20, 181, 41))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(":/images/image/networkx.jpg"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "myApp"))
        self.create_graph.setText(_translate("MainWindow", "生成网络"))
        self.top_router.setText(_translate("MainWindow", "Top10路由器"))
        self.describe_graph.setText(_translate("MainWindow", "统计描述网络"))
        self.addV.setText(_translate("MainWindow", "增加顶点"))
        self.addE.setText(_translate("MainWindow", "增加边"))
        self.change_weight.setText(_translate("MainWindow", "对指定边赋以新的权重"))
        self.routing_table.setText(_translate("MainWindow", "重新生成路由表"))
        self.delE.setText(_translate("MainWindow", "删除边"))
        self.delV.setText(_translate("MainWindow", "删除顶点"))
        self.label.setText(_translate("MainWindow", "plot your network"))
        self.draw_degree.setText(_translate("MainWindow", "绘制节点度分布"))
        self.draw_network.setText(_translate("MainWindow", "绘制网络"))
        self.draw_min_tree.setText(_translate("MainWindow", "绘制最小生成树"))
import res_rc