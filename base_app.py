import sys  # sys нужен для передачи argv в QApplication
import PyQt5
from PyQt5 import QtWidgets
import designer_base as des  # Это наш конвертированный файл дизайна
import db_work as db
import datetime, pytz


class ExampleApp(QtWidgets.QMainWindow, des.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна

        # Ставим конечную дату-время на текущий момент
        now = PyQt5.QtCore.QDateTime.currentDateTime()
        self.end_dt.setDateTime(now)
        # Начальная на день раньше текущего момента
        self.begin_dt.setMaximumDateTime(now)
        self.end_dt.setMinimumDateTime(now)
        now = now.addDays(-1)
        self.begin_dt.setDateTime(now)

        self.text = ""  # Переменная для хранения текста

        self.go_btn.clicked.connect(self.send_request_to_db)  # Устанавливаем в кнопку событие на клик

    def send_request_to_db(self):
        # Передаём запрос в файл для работы с базой данных
        time_b = self.begin_dt.dateTime().toString('yyyy-MM-dd HH:mm:ss')
        time_e = self.end_dt.dateTime().toString('yyyy-MM-dd HH:mm:ss')
        # Времена переведённые в стринги

        # in datetime.datetime.utcnow()
        # 2019-07-31 06:31:17.920689
        # dt = time.toString('yyyy-MM-dd HH:mm:ss')

        local = pytz.timezone("Asia/Krasnoyarsk")
        # Наша тайм-зона

        naive_b = datetime.datetime.strptime(time_b, "%Y-%m-%d %H:%M:%S")
        # В формате datetime
        local_dt_b = local.localize(naive_b, is_dst=None)
        utc_dt_b = local_dt_b.astimezone(pytz.utc)
        # В формате utc, вроде

        naive_e = datetime.datetime.strptime(time_e, "%Y-%m-%d %H:%M:%S")
        local_dt_e = local.localize(naive_e, is_dst=None)
        utc_dt_e = local_dt_e.astimezone(pytz.utc)
        # Если будет ругаться на utc_dt, заливай вместо них native

        type_str = self.type_cb.itemText(self.type_cb.currentIndex())
        # Текст текущего элемента комбобокса, предусмотрено 4 типа
        # Any (когда не важно какого типа запись)
        # И отсальные 3: Input, Output, Unknown
        ret_message = db.dwd_data(naive_b, naive_e, type_str)
        if ret_message is not None:
            self.text += ret_message.__str__() + "\n"
            self.out_text.setText(self.text)
        # Тут просто вызываю функцию в скрипте, она печатает текст и принимает
        # время начальное, конечное и тип. Там напишешь выгрузку уже


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
