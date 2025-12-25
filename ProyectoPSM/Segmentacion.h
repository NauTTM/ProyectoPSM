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
	Mat BalanceBlancos(Mat Imagen);
	struct ObjetosSegmentados {
		vector<Mat> imagenesColor; // I_norm_all
		vector<Mat> mascarasBin;   // BW_norm_all
	};


	vector<Mat> AumentoSaturacion(Mat I_wb);
	vector<Mat> CorreccionIluminacion(vector<Mat> hsv_channels);
	Mat SegmentacionImagen(vector<Mat> hsv_channels);
	Mat FiltrarObjetoLego(Mat mask);
	ObjetosSegmentados RecorteAjusteImagen(vector<Mat> hsv_channels, Mat BW);

	
};

