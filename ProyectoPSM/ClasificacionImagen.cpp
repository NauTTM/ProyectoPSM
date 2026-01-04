#include "ClasificacionImagen.h"

ClasificacionImagen::ClasificacionImagen() {
	ModeloCargado = ml::RTrees::load("datos/clasificador_RF.xml");
	
	// Cargar mu y sigma
	FileStorage fs("datos/parametros_norm.xml", FileStorage::READ);
	Mat muMat, sigmaMat;
	fs["mu"] >> muMat;
	fs["sigma"] >> sigmaMat;
	fs.release();

	VectorMu = (vector<double>) muMat;
	VectorSigma = (vector<double>) sigmaMat;
}
ClasificacionImagen::~ClasificacionImagen() {

}

void ClasificacionImagen::Clasificacion(const vector<double>& caracteristicasVector) {

	vector<double> caracteristicas_normalizadas = NormalizarCaracteristicas(caracteristicasVector);
	int id_clase = Prediccion(caracteristicas_normalizadas);
	emit ResultadoClasificacion(id_clase);
}


vector<double> ClasificacionImagen::NormalizarCaracteristicas(const vector<double>& caracteristicas) {

	vector<double> caracteristicas_normalizadas;

	// 2. Aplicar la normalización (feat - mu) / sigma
	for (size_t i = 0; i < caracteristicas.size(); ++i) {
		// Es vital proteger contra división por cero si algún sigma es 0
		double s = (VectorSigma[i] == 0) ? 1.0 : VectorSigma[i];

		double valor_normalizado = (caracteristicas[i] - VectorMu[i]) / s;
		caracteristicas_normalizadas.push_back(valor_normalizado);
	}
	return caracteristicas_normalizadas;
}

int ClasificacionImagen::Prediccion(const vector<double> &caracteristicas_normalizadas) {
	Mat muestra = Mat(1, caracteristicas_normalizadas.size(), CV_32F);
	for (size_t i = 0; i < caracteristicas_normalizadas.size(); ++i) {
		muestra.at<float>(0, i) = static_cast<float>(caracteristicas_normalizadas[i]);
	}

	// 2. Realizar la predicción
	// predict devuelve un float con el ID de la clase
	float id_clase =ModeloCargado->predict(muestra);
	return (int)id_clase;
}