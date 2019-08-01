# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Work\TensorFlowProject\designer_base.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(339, 566)
        MainWindow.setMinimumSize(QtCore.QSize(339, 566))
        MainWindow.setMaximumSize(QtCore.QSize(339, 566))
        MainWindow.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.go_btn = QtWidgets.QPushButton(self.centralwidget)
        self.go_btn.setGeometry(QtCore.QRect(10, 210, 311, 91))
        self.go_btn.setObjectName("go_btn")
        self.begin_dt = QtWidgets.QDateTimeEdit(self.centralwidget)
        self.begin_dt.setGeometry(QtCore.QRect(140, 10, 181, 41))
        self.begin_dt.setDateTime(QtCore.QDateTime(QtCore.QDate(2019, 7, 1), QtCore.QTime(0, 0, 0)))
        self.begin_dt.setCalendarPopup(True)
        self.begin_dt.setObjectName("begin_dt")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 20, 81, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 80, 81, 31))
        self.label_2.setObjectName("label_2")
        self.end_dt = QtWidgets.QDateTimeEdit(self.centralwidget)
        self.end_dt.setGeometry(QtCore.QRect(140, 70, 181, 41))
        self.end_dt.setDateTime(QtCore.QDateTime(QtCore.QDate(2019, 7, 2), QtCore.QTime(0, 0, 0)))
        self.end_dt.setCalendarPopup(True)
        self.end_dt.setObjectName("end_dt")
        self.type_cb = QtWidgets.QComboBox(self.centralwidget)
        self.type_cb.setGeometry(QtCore.QRect(140, 140, 181, 22))
        self.type_cb.setMaxVisibleItems(3)
        self.type_cb.setObjectName("type_cb")
        self.type_cb.addItem("")
        self.type_cb.addItem("")
        self.type_cb.addItem("")
        self.type_cb.addItem("")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 130, 81, 31))
        self.label_3.setObjectName("label_3")
        self.out_text = QtWidgets.QTextBrowser(self.centralwidget)
        self.out_text.setGeometry(QtCore.QRect(10, 320, 311, 221))
        self.out_text.setObjectName("out_text")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.type_cb.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Database Requests"))
        self.go_btn.setText(_translate("MainWindow", "Search"))
        self.label.setText(_translate("MainWindow", "Begin Time"))
        self.label_2.setText(_translate("MainWindow", "End Time"))
        self.type_cb.setItemText(0, _translate("MainWindow", "Any"))
        self.type_cb.setItemText(1, _translate("MainWindow", "Input"))
        self.type_cb.setItemText(2, _translate("MainWindow", "Output"))
        self.type_cb.setItemText(3, _translate("MainWindow", "Unknown"))
        self.label_3.setText(_translate("MainWindow", "Type"))
