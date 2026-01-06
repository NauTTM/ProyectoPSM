#pragma once
#include <QObject>
#include <QtCore>
#include "opencv2/opencv.hpp"
#include <fstream>

using namespace cv;
using namespace std;

class Segmentacion : public QThread {
	Q_OBJECT

public:
	//constructor por defecto
	Segmentacion();

	//destructor
	~Segmentacion();
	Mat BalanceBlancos(const Mat& Imagen);
	vector<Mat> AumentoSaturacion(const Mat& I_wb);
	vector<Mat> CorreccionIluminacion(const vector<Mat> &hsv_channels);
	Mat SegmentacionImagen(const vector<Mat> &hsv_channels);
	Mat FiltrarObjetoLego(const Mat& mask);
	Mat RecorteAjusteImagen(const vector<Mat> &hsv_channels, const Mat& BW);
	vector<vector<Point>> MostrarBordes(const Mat& bw);
	void SegmentarTodasImagenes();

public slots:
	void SegmentarImagen(const Mat& Imagen);
signals:
	void SegmentacionCompletada(const Mat& Imagen, const vector<vector<Point>>& Bordes);
};