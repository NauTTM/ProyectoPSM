#include "ExtraccionCaracteristicas.h"

ExtraccionCaracteristicas::ExtraccionCaracteristicas() {
	
}

ExtraccionCaracteristicas::~ExtraccionCaracteristicas(void) {
}

void ExtraccionCaracteristicas::ExtraerCaracteristicasImagen(const Mat& ImagenSegmentadaColorTamanoAjustado) {

	//cv::Mat ImagenSegmentadaColorTamanoAjustado = cv::imread("pruebas/01_000_10_001_norm_01.png");

	Mat gray, bw;

	// 1. Convertimos la imagen segmentada a gris
	cvtColor(ImagenSegmentadaColorTamanoAjustado, gray, COLOR_BGR2GRAY);

	// 2. Como ya está segmentada, cualquier píxel > 0 es parte del objeto.
	// Creamos la máscara binaria directamente de los píxeles no negros.
	threshold(gray, bw, 0, 255, THRESH_BINARY | THRESH_OTSU);

	VectorCaracteristicas props{};
	props.rgb = obtenerMedianaMediaRGB(ImagenSegmentadaColorTamanoAjustado);
	props.hu = obtenerMomentosHu(bw);
	props.props = obtenerPropiedadesImagen(bw);

	//emit ListaCaracterisiticas(props);
}


ExtraccionCaracteristicas::RGBResult ExtraccionCaracteristicas::obtenerMedianaMediaRGB(const Mat& ImagenSegmentadaColor) {

	vector<double> R_f, G_f, B_f;
	double thr = 0.98;
	double inv255 = 1.0 / 255.0;

	double sumR = 0, sumG = 0, sumB = 0;
	int countR = 0, countG = 0, countB = 0;


	for (int i = 0; i < ImagenSegmentadaColor.rows; i++) {
		const Vec3b* intensity = ImagenSegmentadaColor.ptr<Vec3b>(i);
		for (int j = 0; j < ImagenSegmentadaColor.cols; j++) {
			const Vec3b& intensityRow = intensity[j];

			if (intensityRow[0] | intensityRow[1] | intensityRow[2]) {
				double b = intensityRow.val[0] * inv255;
				double g = intensityRow.val[1] * inv255;
				double r = intensityRow.val[2] * inv255;

				if (r < thr) { R_f.push_back(r); sumR += r; ++countR; }
				if (g < thr) { G_f.push_back(g); sumG += g; ++countG; }
				if (b < thr) {
					B_f.push_back(b); sumB += b; ++countB;
				}
			}
		}
	}

	auto calcularMediana = [](vector<double>& valores) {
		if (valores.empty()) return 0.0;
		size_t mid = valores.size() / 2;
		nth_element(valores.begin(), valores.begin() + mid, valores.end());
		if (valores.size() % 2 == 0) {
			double a = *max_element(valores.begin(), valores.begin() + mid);
			return (a + valores[mid]) / 2.0;
		} else {
			return valores[mid];
		}
	};

	RGBResult result{};
	result.R_mediana = calcularMediana(R_f);
	result.G_mediana = calcularMediana(G_f);
	result.B_mediana = calcularMediana(B_f);

	result.R_media = countR ? sumR / countR : 0.0;
	result.G_media = countG ? sumG / countG : 0.0;
	result.B_media = countB ? sumB / countB : 0.0;

	return result;
}


ExtraccionCaracteristicas::MomentosHu ExtraccionCaracteristicas::obtenerMomentosHu(const Mat& BW)
{

	// 2. Calcular los momentos espaciales y centrales (m00, m10, c11, etc.)
	Moments mu = moments(BW, false);

	// 3. Calcular los 7 momentos de Hu
	double hu[7];
	HuMoments(mu, hu);
	

	return { hu[0], hu[1], hu[2] };
}

ExtraccionCaracteristicas::ImagenProps ExtraccionCaracteristicas::obtenerPropiedadesImagen(const Mat& BW) {

	// 2. Encontrar contornos (necesario para emular regionprops)
	vector<vector<Point>> contours;
	findContours(BW, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	ImagenProps props = { 0.0, 0.0, 0.0 };

	// 3. Buscar el objeto (idxA en tu MATLAB)
	// MATLAB suele ordenar por área o tomar el objeto principal. 
	// Aquí buscamos el contorno con el área máxima.
	double maxArea = 0;
	int idxA = -1;

	for (int i = 0; i < contours.size(); i++) {
		double areaActual = contourArea(contours[i]);
		if (areaActual > maxArea) {
			maxArea = areaActual;
			idxA = i;
		}
	}

	// 4. Extraer Area, Circularity y Perimeter
	if (idxA != -1) {
		props.area = maxArea;

		// Perimeter
		props.perimetro = arcLength(contours[idxA], true);

		// Circularity: (4 * pi * Area) / (Perimeter^2)
		if (props.perimetro > 0) {
			props.circularidad = (4.0 * CV_PI * props.area) / pow(props.perimetro, 2);
		}
	}

	return props;
}

