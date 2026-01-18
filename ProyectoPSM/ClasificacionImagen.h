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
	~ClasificacionImagen();
private:
	vector<double> VectorSigma;
	vector<double> VectorMu;
	Ptr<ml::RTrees> ModeloCargado;

	vector<double> ExtraerCaracteristicasImagen(const vector<Mat>& ImagenSegmentadaColorTamanoAjustado);
	vector<vector<double>> NormalizarCaracteristicas(const vector<vector<double>>& vector);
	vector<int> Prediccion(const vector<vector<double>>& caracteristicas_normalizadas);
	
public slots:
	void Clasificacion(const vector<vector<double>>& caracteristicasVector);

signals:
	void ResultadoClasificacion(const vector<int> id_clase);
};