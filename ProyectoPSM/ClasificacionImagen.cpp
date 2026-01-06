#include "ClasificacionImagen.h"

// Constructor
ClasificacionImagen::ClasificacionImagen() {
	ModeloCargado = ml::RTrees::load("datos/clasificador_RF.xml"); // Carg del modelo Random Forest entrenador
	
	// Cargar de los parametros de normalizacion (mu y sigma)
	FileStorage fs("datos/parametros_norm.xml", FileStorage::READ);
	Mat muMat, sigmaMat;
	fs["mu"] >> muMat;
	fs["sigma"] >> sigmaMat;
	fs.release();

	// Conversion a vectores estandar de C++
	VectorMu = (vector<double>) muMat;
	VectorSigma = (vector<double>) sigmaMat;
}
ClasificacionImagen::~ClasificacionImagen() {

}

// Funcionc clasificacion
void ClasificacionImagen::Clasificacion(const vector<double>& caracteristicasVector) {

	vector<double> caracteristicas_normalizadas = NormalizarCaracteristicas(caracteristicasVector); // Normalizacion de las caracteristicas
	int id_clase = Prediccion(caracteristicas_normalizadas); // Prediccion de la clase
	emit ResultadoClasificacion(id_clase); // Emision del resultado
}

// Funcion normalizar caracteristicas
vector<double> ClasificacionImagen::NormalizarCaracteristicas(const vector<double>& caracteristicas) {

	// Inicializacion del vector de salida
	vector<double> caracteristicas_normalizadas;

	// Aplicar la normalización (feat - mu) / sigma
	for (size_t i = 0; i < caracteristicas.size(); ++i) {
		// Proteccion contra division por cero si algun sigma es 0
		double s = (VectorSigma[i] == 0) ? 1.0 : VectorSigma[i];

		// Formula de normalizacion
		double valor_normalizado = (caracteristicas[i] - VectorMu[i]) / s;
		// Almacenamiento del resultado
		caracteristicas_normalizadas.push_back(valor_normalizado);
	}

	// Retorno
	return caracteristicas_normalizadas;
}

//Funcion Prediccion
int ClasificacionImagen::Prediccion(const vector<double> &caracteristicas_normalizadas) {
	// Preparacion de la meustra de entrada
	Mat muestra = Mat(1, caracteristicas_normalizadas.size(), CV_32F);

	// Copia del vector al formato OpenCV
	for (size_t i = 0; i < caracteristicas_normalizadas.size(); ++i) {
		muestra.at<float>(0, i) = static_cast<float>(caracteristicas_normalizadas[i]);
	}

	// sRealizar la predicción
	// predict devuelve un float con el ID de la clase
	float id_clase =ModeloCargado->predict(muestra);
	// Retorno
	return (int)id_clase;
}