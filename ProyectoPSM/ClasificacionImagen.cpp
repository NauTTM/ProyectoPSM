#include "ClasificacionImagen.h"

// Constructor
ClasificacionImagen::ClasificacionImagen() {
	ModeloCargado = ml::RTrees::load("datos/PropsRGBHS17/clasificador_RF.xml"); // Carg del modelo Random Forest entrenador
	
	// Cargar de los parametros de normalizacion (mu y sigma)
	FileStorage fs("datos/PropsRGBHS17/parametros_norm.xml", FileStorage::READ);
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
void ClasificacionImagen::Clasificacion(const vector<vector<double>>& caracteristicasVector) {

	vector<vector<double>> caracteristicas_normalizadas = NormalizarCaracteristicas(caracteristicasVector); // Normalizacion de las caracteristicas
	vector<int> id_clase = Prediccion(caracteristicas_normalizadas); // Prediccion de la clase

	emit ResultadoClasificacion(id_clase); // Emision del resultado
}

// Funcion normalizar caracteristicas
vector<vector<double>> ClasificacionImagen::NormalizarCaracteristicas(const vector<vector<double>>& caracteristicas) {

	// Inicializacion del vector de salida
	vector<vector<double>> caracteristicas_normalizadas;
	caracteristicas_normalizadas.resize(caracteristicas.size());
	// Aplicar la normalización (feat - mu) / sigma
	for (size_t i = 0; i < caracteristicas.size(); ++i) {
		for(int j = 0; j < caracteristicas[i].size(); j++){
		// Proteccion contra division por cero si algun sigma es 0
		double s = (VectorSigma[j] == 0) ? 1.0 : VectorSigma[j];

		// Formula de normalizacion
		double valor_normalizado = (caracteristicas[i][j] - VectorMu[j]) / s;
		// Almacenamiento del resultado
		caracteristicas_normalizadas[i].push_back(valor_normalizado);
		}
	}

	// Retorno
	return caracteristicas_normalizadas;
}

//Funcion Prediccion
vector<int> ClasificacionImagen::Prediccion(const vector<vector<double>> &caracteristicas_normalizadas) {
	vector<int> ids_clase;
	for (int j = 0; j < caracteristicas_normalizadas.size(); j++) {
		// Preparacion de la meustra de entrada
		Mat muestra = Mat(1, caracteristicas_normalizadas[j].size(), CV_32F);

		// Copia del vector al formato OpenCV
		for (size_t i = 0; i < caracteristicas_normalizadas[j].size(); ++i) {
			muestra.at<float>(0, i) = static_cast<float>(caracteristicas_normalizadas[j][i]);
		}

		// sRealizar la predicción
		// predict devuelve un float con el ID de la clase
		float id_clase = ModeloCargado->predict(muestra);
		ids_clase.push_back((int)id_clase);
	}
	// Retorno
	return ids_clase;
}