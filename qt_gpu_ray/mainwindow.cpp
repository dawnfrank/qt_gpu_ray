#include <QtWidgets\qlabel.h>
#include <QtGui\qimage.h>
#include <QtGui\qpixmap.h>
#include <QtCore\qtimer.h>
#include <QtGui\qevent.h>

#include "define.h"
#include "mainwindow.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdio.h"

extern "C" World* InitWorld(const int w, const int h);
extern "C" void ModifyWorld(SceneData sData, World* dev_world);
extern "C" void RenderWorld(const int w, const int h, unsigned char *dev_bitmap, unsigned char *host_bitmap, World* dev_world);


MainWindow::MainWindow() {
	ui.setupUi(this);
	setFocusPolicy(Qt::StrongFocus);

	label = new QLabel(ui.widget);
	label->setFixedSize(size());

	timer = new QTimer(ui.widget);
	connect(timer, &QTimer::timeout, this, &MainWindow::render_scene);
	timer->start(20);

	const int imageSize = w*h * 4;
	host_bitmap = new unsigned char[imageSize];
	cudaMalloc(&dev_bitmap, imageSize);

	dev_world = InitWorld(w, h);
}

void MainWindow::render_scene()
{
	fps.Update();
	ModifyWorld(sData, dev_world);
	sData.reset();
	RenderWorld(w, h, dev_bitmap, host_bitmap, dev_world);
	QImage image(host_bitmap, w, h, QImage::Format_ARGB32);
	label->setPixmap(QPixmap::fromImage(image));
}

void MainWindow::keyPressEvent(QKeyEvent * event)
{
	if (event->key() == Qt::Key_W) sData.cameraCmd |= CAMERA_CMD_W;
	else if (event->key() == Qt::Key_S) sData.cameraCmd |= CAMERA_CMD_S;
	else if (event->key() == Qt::Key_A) sData.cameraCmd |= CAMERA_CMD_A;
	else if (event->key() == Qt::Key_D) sData.cameraCmd |= CAMERA_CMD_D;
}

void MainWindow::mousePressEvent(QMouseEvent * event)
{
	mouseDownPos.x = event->pos().x();
	mouseDownPos.y = event->pos().y();
}

void MainWindow::mouseMoveEvent(QMouseEvent * event)
{
	int x = event->pos().x();
	int y = event->pos().y();
	if (event->buttons() & Qt::RightButton) {
		sData.cameraRotateY = x - mouseDownPos.x;
		sData.cameraRotateX = y - mouseDownPos.y;

		mouseDownPos.x = x;
		mouseDownPos.y = y;
	}
}

void MainWindow::wheelEvent(QWheelEvent * event)
{
	sData.cameraZoom = -event->delta();
}
