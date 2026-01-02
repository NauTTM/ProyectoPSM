#pragma once
#include <QObject>
#include <QtCore>
#include "opencv2/opencv.hpp"
#include "ExtraccionCaracteristicas.h"

using namespace cv;
using namespace std;

class ClasificacionImagen : public QThread {
	Q_OBJECT
public: 
	ClasificacionImagen();
	void Clasificacion(const vector<double>& caracteristicasVector);
private:
	struct Modelo {
		Ptr<ml::RTrees> modeloCargado;
		Mat muMat;
		Mat sigmaMat;
	};
	Modelo CargarModelo();
	ExtraccionCaracteristicas::VectorCaracteristicas ExtraerCaracteristicasImagen(const Mat& ImagenSegmentadaColorTamanoAjustado);
	vector<double> PrepararVector(const ExtraccionCaracteristicas::VectorCaracteristicas& vector);
	vector<double> NormalizarCaracteristicas(const vector<double>& mu, const vector<double>& sigma, const vector<double>& vector);
	float Prediccion(const vector<double>& caracteristicas_normalizadas, const Ptr<ml::RTrees> modeloCargado);
};