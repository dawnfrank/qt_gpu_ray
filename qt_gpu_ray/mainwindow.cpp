#include <QtWidgets\qlabel.h>
#include <QtGui\qimage.h>
#include <QtGui\qpixmap.h>

#include "mainwindow.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C" cudaError_t InitCuda(const int w, const int h, unsigned char **dev_bitmap);
extern "C" cudaError_t CalculateCuda(const int w, const int h, unsigned char *dev_bitmap, unsigned char *host_bitmap);


MainWindow::MainWindow() {
	ui.setupUi(this);

	label = new QLabel(ui.widget);

	host_bitmap = new unsigned char[w*h * 4];
	InitCuda(w, h, &dev_bitmap);

	render_scene();
}

void MainWindow::render_scene()
{
	CalculateCuda(w, h, dev_bitmap, host_bitmap);
	QImage image(host_bitmap, w, h, QImage::Format_ARGB32);
	label->setPixmap(QPixmap::fromImage(image));
}
