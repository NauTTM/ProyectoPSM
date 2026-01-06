#include "ExtraccionCaracteristicas.h"

ExtraccionCaracteristicas::ExtraccionCaracteristicas() {
	
}

ExtraccionCaracteristicas::~ExtraccionCaracteristicas(void) {
}

// Funcion extraer caracteristicas
void ExtraccionCaracteristicas::ExtraerCaracteristicasImagen(const Mat& ImagenSegmentadaColorTamanoAjustado) {

	// Variables
	Mat gray, bw;

	// Convertimos la imagen segmentada a gris
	cvtColor(ImagenSegmentadaColorTamanoAjustado, gray, COLOR_BGR2GRAY);

	// Como ya esta segmentada, cualquier píxel > 0 es parte del objeto.
	// Creamos la mascara binaria directamente de los pixeles no negros.
	threshold(gray, bw, 0, 255, THRESH_BINARY | THRESH_OTSU);

	// Extraccion de caracteristicas por bloques
	vector<double> vectorCaracteristicas, rgb, hu, props; 
	rgb = obtenerMedianaMediaRGB(ImagenSegmentadaColorTamanoAjustado); // Caracteristicas de color
	hu = obtenerMomentosHu(bw); // Caracteristicas de forma (Momentos de Hu)
	props = obtenerPropiedadesImagen(bw); // Propiedades geometricas

	// Construccion del vector final
	vectorCaracteristicas.insert(vectorCaracteristicas.end(), rgb.begin(), rgb.end());
	vectorCaracteristicas.insert(vectorCaracteristicas.end(), hu.begin(), hu.end());
	vectorCaracteristicas.insert(vectorCaracteristicas.end(), props.begin(), props.end());
	
	// Emision del resultado
	emit ListaCaracterisiticas(vectorCaracteristicas);
}

// Funcion obetner mediana media RGB
vector<double> ExtraccionCaracteristicas::obtenerMedianaMediaRGB(const Mat& ImagenSegmentadaColor) {

	// Inicializacion de estructuras y parametros
	vector<double> R_f, G_f, B_f;
	double thr = 0.98; // Umbral para descartar pixeles saturados
	double inv255 = 1.0 / 255.0;

	// Acumuladores para la media
	double sumR = 0, sumG = 0, sumB = 0;
	int countR = 0, countG = 0, countB = 0;

	// Recorrido pixel a pixel
	for (int i = 0; i < ImagenSegmentadaColor.rows; i++) {
		const Vec3b* intensity = ImagenSegmentadaColor.ptr<Vec3b>(i);
		for (int j = 0; j < ImagenSegmentadaColor.cols; j++) {
			const Vec3b& intensityRow = intensity[j];

			// Exclusion del fondo comprobando si el pixel no es negro
			if (intensityRow[0] | intensityRow[1] | intensityRow[2]) {
				//Normalizacion del color
				double b = intensityRow.val[0] * inv255;
				double g = intensityRow.val[1] * inv255;
				double r = intensityRow.val[2] * inv255;

				// Rechazo de valores saturados y acumulacion de datos para la mediana
				if (r < thr) { R_f.push_back(r); sumR += r; ++countR; } 
				if (g < thr) { G_f.push_back(g); sumG += g; ++countG; }
				if (b < thr) {
					B_f.push_back(b); sumB += b; ++countB;
				}
			}
		}
	}

	// Caluclo de la mediana
	auto calcularMediana = [](vector<double>& valores) {
		if (valores.empty()) return 0.0;
		size_t mid = valores.size() / 2;
		nth_element(valores.begin(), valores.begin() + mid, valores.end());
		// Mediana para numero par
		if (valores.size() % 2 == 0) {
			double a = *max_element(valores.begin(), valores.begin() + mid);
			return (a + valores[mid]) / 2.0;
		} else {
			return valores[mid];
		}
	};

	// Calculo de medianas
	double R_mediana = calcularMediana(R_f);
	double G_mediana = calcularMediana(G_f);
	double B_mediana = calcularMediana(B_f);

	// Calculo de medias finales con proteccion para divisiones entre cero
	double R_media = countR ? sumR / countR : 0.0;
	double G_media = countG ? sumG / countG : 0.0;
	double B_media = countB ? sumB / countB : 0.0;

	// Vector de salida
	return { R_mediana, G_mediana, B_mediana, R_media, G_media, B_media };
}


vector<double> ExtraccionCaracteristicas::obtenerMomentosHu(const Mat& BW)
{

	// Calcular los momentos espaciales y centrales (m00, m10, c11, etc.)
	Moments mu = moments(BW, false);

	// Calcular los 7 momentos de Hu
	double hu[7];
	HuMoments(mu, hu);
	
	// Enviamos los momentos de Hu mas relevantes y diferenciables entre clases
	return { hu[0], hu[1], hu[2] };
}

// Funcion obtener propiedades de la imagen
vector<double> ExtraccionCaracteristicas::obtenerPropiedadesImagen(const Mat& BW) {

	// Inicializacion de variables
	double area = 0, perimetro = 0, circularidad = 0;

	// Deteccion de contornos contornos (necesario para emular regionprops)
	vector<vector<Point>> contours;
	findContours(BW, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// Buscar el objeto principal 

	double maxArea = 0;
	int idxA = -1;

	for (int i = 0; i < contours.size(); i++) {
		double areaActual = contourArea(contours[i]);
		if (areaActual > maxArea) {
			maxArea = areaActual;
			idxA = i;
		}
	}

	// Extraer Area, Circularity y Perimeter
	if (idxA != -1) {
		area = maxArea;

		// Perimeter
		perimetro = arcLength(contours[idxA], true);

		// Circularity: (4 * pi * Area) / (Perimeter^2)
		if (perimetro > 0) {
			circularidad = (4.0 * CV_PI * area) / pow(perimetro, 2);
		}
	}

	// Vector de salida
	return { area, perimetro, circularidad };
}

