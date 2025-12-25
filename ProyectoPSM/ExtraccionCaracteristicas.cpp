#include "ExtraccionCaracteristicas.h"

ExtraccionCaracteristicas::ExtraccionCaracteristicas() {
	
}

ExtraccionCaracteristicas::~ExtraccionCaracteristicas(void) {
}

void ExtraccionCaracteristicas::listaCaracteristicas() {

	cv::Mat ImagenSegmentadaColor = cv::imread("pruebas/01_000_10_001.jpg_obj.png");
	cv::Mat ImagenSegmentadaColorTamanoAjustado = cv::imread("pruebas/01_000_10_001_norm_01.png");

	VectorCaracteristicas props{};
	props.rgb = obtenerMedianaMediaRGB(ImagenSegmentadaColor);
	props.hu = obtenerMomentosHu(ImagenSegmentadaColor);
	props.props = obtenerPropiedadesImagen(ImagenSegmentadaColorTamanoAjustado);
}


ExtraccionCaracteristicas::RGBResult ExtraccionCaracteristicas::obtenerMedianaMediaRGB(const cv::Mat& ImagenSegmentadaColor) {

	std::vector<double> R_f, G_f, B_f;
	double thr = 0.98;


	
	for (int i = 0; i < ImagenSegmentadaColor.rows; i++) {
		for (int j = 0; j < ImagenSegmentadaColor.cols; j++) {
			cv::Vec3b intensity = ImagenSegmentadaColor.at<cv::Vec3b>(i, j);
			if (intensity[0] > 0 || intensity[1] > 0 || intensity[2] > 0) {
				double b = intensity.val[0] / 255.0;
				double g = intensity.val[1] / 255.0;
				double r = intensity.val[2] / 255.0;

				if (r < thr) R_f.push_back(r);
				if (g < thr) G_f.push_back(g);
				if (b < thr) B_f.push_back(b);
			}
		}
	}

	auto calcularMediana = [](std::vector<double>& valores) {
		if (valores.empty()) return 0.0;
		std::sort(valores.begin(), valores.end());
		size_t mid = valores.size() / 2;
		if (valores.size() % 2 == 0) {
			return (valores[mid - 1] + valores[mid]) / 2.0;
		} else {
			return valores[mid];
		}
	};

	auto calcularMedia = [](std::vector<double>& valores) {
		if (valores.empty()) return 0.0;
		double suma = std::accumulate(valores.begin(), valores.end(), 0.0);
		return suma / valores.size();
	};

	RGBResult result{};
	result.R_mediana = calcularMediana(R_f);
	result.G_mediana = calcularMediana(G_f);
	result.B_mediana = calcularMediana(B_f);

	result.R_media = calcularMedia(R_f);
	result.G_media = calcularMedia(G_f);
	result.B_media = calcularMedia(B_f);

	return result;
}


ExtraccionCaracteristicas::MomentosHu ExtraccionCaracteristicas::obtenerMomentosHu(const cv::Mat& ImagenSegmentadaColor)
{
	cv::Mat gray, bw;

	// 1. Convertimos la imagen segmentada a gris
	cv::cvtColor(ImagenSegmentadaColor, gray, cv::COLOR_BGR2GRAY);

	// 2. Como ya está segmentada, cualquier píxel > 0 es parte del objeto.
	// Creamos la máscara binaria directamente de los píxeles no negros.
	cv::threshold(gray, bw, 0, 255, cv::THRESH_BINARY);

	// 2. Calcular los momentos espaciales y centrales (m00, m10, c11, etc.)
	cv::Moments mu = cv::moments(bw, false);

	// 3. Calcular los 7 momentos de Hu
	double hu[7];
	cv::HuMoments(mu, hu);
	

	return { hu[0], hu[1], hu[2] };
}

ExtraccionCaracteristicas::ImagenProps ExtraccionCaracteristicas::obtenerPropiedadesImagen(const cv::Mat& ImagenSegmentadaColorTamanoAjustado) {

	cv::Mat gray, bw;

	// 1. BW = imbinarize(rgb2gray(I));
	cv::cvtColor(ImagenSegmentadaColorTamanoAjustado, gray, cv::COLOR_BGR2GRAY);
	// THRESH_OTSU calcula automáticamente el umbral igual que imbinarize
	cv::threshold(gray, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	// 2. Encontrar contornos (necesario para emular regionprops)
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(bw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	ImagenProps props = { 0.0, 0.0, 0.0 };

	// 3. Buscar el objeto (idxA en tu MATLAB)
	// MATLAB suele ordenar por área o tomar el objeto principal. 
	// Aquí buscamos el contorno con el área máxima.
	double maxArea = 0;
	int idxA = -1;

	for (int i = 0; i < contours.size(); i++) {
		double areaActual = cv::contourArea(contours[i]);
		if (areaActual > maxArea) {
			maxArea = areaActual;
			idxA = i;
		}
	}

	// 4. Extraer Area, Circularity y Perimeter
	if (idxA != -1) {
		props.area = maxArea;

		// Perimeter
		props.perimetro = cv::arcLength(contours[idxA], true);

		// Circularity: (4 * pi * Area) / (Perimeter^2)
		if (props.perimetro > 0) {
			props.circularidad = (4.0 * CV_PI * props.area) / std::pow(props.perimetro, 2);
		}
	}

	return props;
}

