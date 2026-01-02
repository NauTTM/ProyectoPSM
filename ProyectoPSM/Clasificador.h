#pragma once
#include <QObject>
#include <QtCore>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

class Clasificador : public QThread {
	Q_OBJECT
public:
	Clasificador();
	void Clasificador_RF();
private:
	struct ParamsNormalizacion {
		vector<double> mu;
		vector<double> sigma;
		vector<vector<double>> Xn;
	};
	struct KFoldPartition {
		vector<vector<int>> trainIndices;
		vector<vector<int>> testIndices;
	};
	vector<vector<double>> CargarObservacionesCSV(QString rutaArchivo);
	vector<double> CargarEtiquetasCSV(QString nombreArchivo);
	ParamsNormalizacion NormalizarDatos(const vector<vector<double>>& X);
	KFoldPartition CrearCVPartition(int numMuestras, int K);
	Ptr<ml::RTrees> EntrenarRandomForest(const vector<vector<double>>& X, const vector<double>& Y);
};
