#!/bin/python3
import sys


from search_engine import SearchEngine


from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QComboBox, QSpinBox


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Search Engine'
        self.k = 10
        self.se = SearchEngine()
        layout = QVBoxLayout()

        self.line_edit = QLineEdit()
        self.line_edit.returnPressed.connect(self.query_entered)
        layout.addWidget(self.line_edit)

        self.mat_selector = QComboBox()
        self.mat_selector.addItems(["SVD","IDF","NOIDF"])
        self.mat_selector.currentTextChanged.connect(self.matrix_changed)
        layout.addWidget(self.mat_selector)

        self.spinbox = QSpinBox()
        self.spinbox.setRange(0, self.se.get_max_vectors())
        self.spinbox.setValue(self.se.get_max_vectors())
        self.spinbox.valueChanged.connect(self.vectors_changed)
        layout.addWidget(self.spinbox)

        self.results = [QLabel() for i in range(self.k)]
        for label in self.results:
            label.setOpenExternalLinks(True)
            layout.addWidget(label)
        self.setLayout(layout)
        self.set_window_title()
        self.show()

    def set_window_title(self):
        m_type = self.mat_selector.currentText()
        vectors = self.spinbox.value()
        if m_type == "SVD":
            self.setWindowTitle(f"{self.title} SVD({vectors})")
        else:
            self.setWindowTitle(f"{self.title} {m_type}")

    def matrix_changed(self, s: str):
        self.se.set_type(s)
        self.spinbox.setRange(0, self.se.get_max_vectors())
        self.spinbox.setValue(self.se.get_max_vectors())
        self.set_window_title()

    def query_entered(self):
        for label in self.results:
            label.setText("")
        results = self.se.search_string(self.line_edit.text(), self.k)
        if isinstance(results, str):
            self.results[0].setText(results)
            return
        for i, result in enumerate(results):
            url, score = result
            self.results[i].setText(f"<b>{score:0.4}</b>    <a href=\"{url}\">{url}</a>")

    def vectors_changed(self, i: int):
        self.se.set_vectors_used(i)
        self.set_window_title()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())