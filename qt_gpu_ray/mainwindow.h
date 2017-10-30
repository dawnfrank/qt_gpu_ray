#pragma once

#include <QtWidgets\qwidget.h>
#include "ui_mainwindow.h"

class QLabel;

class MainWindow :public QWidget {
public:
	MainWindow();
	void render_scene();
private:
	Ui::Form ui;

	QLabel *label;
	const int w = 800;
	const int h = 600;
	unsigned char *host_bitmap;
	unsigned char *dev_bitmap;
};