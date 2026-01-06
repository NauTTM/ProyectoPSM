#pragma once
#include <QObject>
#include <QtCore>
#include "opencv2/opencv.hpp"
#include <fstream>

using namespace cv;
using namespace std;

// Definion de la clase segmentacion en hilo dedicado
class Segmentacion : public QThread {
	Q_OBJECT

public:
	// Constructor por defecto
	Segmentacion();

	// Destructor
	~Segmentacion();

	// Balance de blacos
	Mat BalanceBlancos(const Mat& Imagen);

	// Aumento de saturacion
	vector<Mat> AumentoSaturacion(const Mat& I_wb);

	// Correccion de iluminacion
	vector<Mat> CorreccionIluminacion(const vector<Mat> &hsv_channels);

	// Segmentacion binaria
	Mat SegmentacionImagen(const vector<Mat> &hsv_channels);

	// Filtrado del objetivo principal (LEGO)
	Mat FiltrarObjetoLego(const Mat& mask);

	// Recorte y normalizacion
	Mat RecorteAjusteImagen(const vector<Mat> &hsv_channels, const Mat& BW);

	// Obtencion de bordes
	vector<vector<Point>> MostrarBordes(const Mat& bw);

// Punto de entrada del hilo
public slots:
	void SegmentarImagen(const Mat& Imagen);

// Salidad del proceso de segmentacion
signals:
	void SegmentacionCompletada(const Mat& Imagen, const vector<vector<Point>>& Bordes);
};