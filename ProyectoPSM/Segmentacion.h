#pragma once
#include <QObject>
#include <QtCore>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class Segmentacion : public QThread {
	Q_OBJECT

public:
	//constructor por defecto
	Segmentacion();

	//destructor
	~Segmentacion();
	struct ObjetosSegmentados {
		vector<Mat> imagenesColor; // I_norm_all
		vector<Mat> mascarasBin;   // BW_norm_all
	};
	Mat RedimensionarImagen(const Mat& Imagen);
	Mat BalanceBlancos(const Mat& Imagen);
	vector<Mat> AumentoSaturacion(const Mat& I_wb);
	vector<Mat> CorreccionIluminacion(const vector<Mat> &hsv_channels);
	Mat SegmentacionImagen(const vector<Mat> &hsv_channels);
	Mat FiltrarObjetoLego(const Mat& mask);
	ObjetosSegmentados RecorteAjusteImagen(const vector<Mat> &hsv_channels, const Mat& BW);
	vector<vector<Point>> MostrarBordes(const Mat& bw);
	

public slots:
	void SegmentarImagen(const Mat& Imagen);
signals:
	void SegmentacionCompletada(const Mat& Imagen, const vector<vector<Point>>& Bordes);
};

