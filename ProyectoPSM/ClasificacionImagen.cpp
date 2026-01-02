#include "ClasificacionImagen.h"

ClasificacionImagen::ClasificacionImagen() {
	
}

void ClasificacionImagen::Clasificacion(const vector<double>& caracteristicasVector) {

	Modelo modelo = CargarModelo();
	vector<double> sigma_vec = (vector<double>)modelo.sigmaMat;
	vector<double> mu_vec = (vector<double>)modelo.muMat;
	vector<double> caracteristicas_normalizadas = NormalizarCaracteristicas(mu_vec, sigma_vec, caracteristicasVector);
	float id_clase = Prediccion(caracteristicas_normalizadas, modelo.modeloCargado);
}

ClasificacionImagen::Modelo ClasificacionImagen::CargarModelo() {
	Ptr<ml::RTrees> modeloCargado = ml::RTrees::load("datos/clasificador_RF.xml");

	// Cargar mu y sigma
	FileStorage fs("datos/parametros_norm.xml", FileStorage::READ);
	Mat muMat, sigmaMat;
	fs["mu"] >> muMat;
	fs["sigma"] >> sigmaMat;
	fs.release();

	return { modeloCargado, muMat, sigmaMat };
}

vector<double> ClasificacionImagen::NormalizarCaracteristicas(const vector<double>& mu, const vector<double>& sigma, const vector<double>& caracteristicas) {

	vector<double> caracteristicas_normalizadas;

	// 2. Aplicar la normalización (feat - mu) / sigma
	for (size_t i = 0; i < caracteristicas.size(); ++i) {
		// Es vital proteger contra división por cero si algún sigma es 0
		double s = (sigma[i] == 0) ? 1.0 : sigma[i];

		double valor_normalizado = (caracteristicas[i] - mu[i]) / s;
		caracteristicas_normalizadas.push_back(valor_normalizado);
	}
	return caracteristicas_normalizadas;
}

vector<double> ClasificacionImagen::PrepararVector(const ExtraccionCaracteristicas::VectorCaracteristicas& vector) {

	return {
		vector.rgb.R_mediana, vector.rgb.G_mediana, vector.rgb.B_mediana,
		vector.rgb.R_media,   vector.rgb.G_media,   vector.rgb.B_media,
		vector.hu.hu1,        vector.hu.hu2,        vector.hu.hu3,
		vector.props.area,    vector.props.circularidad, vector.props.perimetro
	};
}

float ClasificacionImagen::Prediccion(const vector<double> &caracteristicas_normalizadas,const Ptr<ml::RTrees> modeloCargado) {
	Mat muestra = Mat(1, caracteristicas_normalizadas.size(), CV_32F);
	for (size_t i = 0; i < caracteristicas_normalizadas.size(); ++i) {
		muestra.at<float>(0, i) = static_cast<float>(caracteristicas_normalizadas[i]);
	}

	// 2. Realizar la predicción
	// predict devuelve un float con el ID de la clase
	float id_clase = modeloCargado->predict(muestra);
	return id_clase;
}