#pragma once
#include <QObject>
#include <QtCore>
#include "opencv2/opencv.hpp"
#include <fstream>

using namespace cv;
using namespace std;

class Orientacion : public QThread {
	Q_OBJECT
public:
	Orientacion();
	struct ModeloReferencia {
		int id_clase;
		int elevacion;
		vector<double> signatura;
	};
	double DeterminarOrientacionTotal(const int id_clase, const Mat& imgColor, const Mat& imgBin, const vector<ModeloReferencia>& baseDatos);
public slots:
	void CalcularOrientacion(const vector<int> id_clase);
	void GuardarImagenesSegmentadas(const vector<Mat>& Imagen, const vector<vector<Point>>& Bordes, const vector<Mat> ImagenesSegmentadasBinarias, const vector<Mat> ImagenesBinariasAColor);
	void CalcularTodasSignaturas();
private:
	vector<Mat> ImgColor, ImgBin;
	vector<ModeloReferencia> BaseDatos;
	vector<ModeloReferencia> CargarReferenciasCSV(string path);
	vector<double> ObtenerSignatura(const Mat& BW);
	void NormalizarSignatura(vector<double>& sig);
	int CalcularDesplazamientoCircular(const vector<double>& sig_actual, const vector<double>& sig_ref);
	bool CalcularAsimetria(const Mat& imgColor, const Mat& imgBin, Point2f centroide);
	ModeloReferencia EncontrarMejorClase(const int id_clase, const vector<double>& sigActual, const vector<ModeloReferencia>& baseDatos);
	double CalcularCorrelacionMaxima(const vector<double>& sig1, const vector<double>& sig2);
	bool ObtenerReferenciaHue(const Mat& imgHSV, const Mat& imgBin, Point2f centroide);
signals:
	void OrientacionCalculada(const vector<int> id_clase, const vector<int> anguloFinal);
};