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
 
private:
	vector<double> obtenerMedianaMediaRGB(const Mat& ImagenSegmentadaColor);
	vector<double> obtenerMomentosHu(const Mat& BW);
	vector<double> obtenerPropiedadesImagen(const Mat& BW);

public slots:
	void ExtraerCaracteristicasImagen(const Mat& ImagenSegmentadaColorTamanoAjustado);

signals:
	void ListaCaracterisiticas(const vector<double> &caracteristicas);
	
};