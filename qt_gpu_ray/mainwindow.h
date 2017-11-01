#pragma once

#include <QtWidgets\qwidget.h>
#include "ui_mainwindow.h"
#include "fps.h"
#include "scenedata.h"

class QLabel;
class QTimer;
class World;

class MainWindow :public QWidget {
public:
	MainWindow();
	void render_scene();
protected:
	void keyPressEvent(QKeyEvent *event) override;
	void mousePressEvent(QMouseEvent *event) override;
	void mouseMoveEvent(QMouseEvent *event) override;
	void wheelEvent(QWheelEvent *event) override;
private:
	Ui::widget_2 ui;

	FPS fps;
	QLabel *label;
	QTimer *timer;

	const int w = 800;
	const int h = 600;
	SceneData sData;
	Pos mouseDownPos;

	unsigned char *host_bitmap;
	unsigned char *dev_bitmap;
	World* dev_world;
};