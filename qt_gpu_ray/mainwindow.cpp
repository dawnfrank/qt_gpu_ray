#include <QtWidgets\qlabel.h>
#include <QtGui\qimage.h>
#include <QtGui\qpixmap.h>
#include <QtCore\qtimer.h>

#include "mainwindow.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdio.h"

extern "C" World* InitWorld(const int w, const int h);
extern "C" void RenderWorld(const int w, const int h, unsigned char *dev_bitmap, unsigned char *host_bitmap, World* dev_world);


MainWindow::MainWindow() {
	ui.setupUi(this);

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
	RenderWorld(w, h, dev_bitmap, host_bitmap, dev_world);
	QImage image(host_bitmap, w, h, QImage::Format_ARGB32);
	label->setPixmap(QPixmap::fromImage(image));
}
