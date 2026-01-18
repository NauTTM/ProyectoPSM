#pragma once
#include <QObject>
#include <QtCore>
#include "opencv2/opencv.hpp"
#include <fstream>

using namespace cv;
using namespace std;

class ExtraccionCaracteristicas : public QThread {
	Q_OBJECT

public:
	ExtraccionCaracteristicas();
	~ExtraccionCaracteristicas(void);
	void ExtraerXyGClasificacion();
private:
	vector<double> obtenerMedianaMediaRGB(const Mat& ImagenSegmentadaColor);
	vector<double> obtenerMomentosHu(const Mat& BW);
	vector<double> obtenerPropiedadesImagen(const Mat& BW);
	vector<double> obtenerMedianaMediaHS(const Mat& ImagenSegmentadaColor);
	vector<double> ExtraerCaracteristicasImagen1(const Mat& ImagenSegmentadaColorTamanoAjustado);
	double obtenerRugosidad(const Mat& BW, const Mat& gray);

public slots:
	void ExtraerCaracteristicasImagen(const vector<Mat>& ImagenSegmentadaColorTamanoAjustado);

signals:
	void ListaCaracterisiticas(const vector<vector<double>> &caracteristicas);
	
};