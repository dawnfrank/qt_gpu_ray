#pragma once

#include <QtWidgets\qwidget.h>
#include "ui_mainwindow.h"
#include "fps.h"

class QLabel;
class QTimer;
class World;

class MainWindow :public QWidget {
public:
	MainWindow();
	void render_scene();
private:
	Ui::widget_2 ui;

	FPS fps;
	QLabel *label;
	QTimer *timer;

	const int w = 800;
	const int h = 600;
	unsigned char *host_bitmap;
	unsigned char *dev_bitmap;
	World* dev_world;
};