#include "ExtraccionCaracteristicas.h"

ExtraccionCaracteristicas::ExtraccionCaracteristicas() {
	
}

ExtraccionCaracteristicas::~ExtraccionCaracteristicas(void) {
}

// Funcion extraer caracteristicas
void ExtraccionCaracteristicas::ExtraerCaracteristicasImagen(const vector<Mat>& ImagenSegmentadaColorTamanoAjustado) {

	
	vector<vector<double>> vector_final;
	for (int i = 0; i < ImagenSegmentadaColorTamanoAjustado.size(); i++) {
		// Variables
		Mat gray, bw;

		// Extraccion de caracteristicas por bloques
		vector<double> vectorCaracteristicas, rgb, hu, props, hs;
		double rugosidad;
		// Convertimos la imagen segmentada a gris
		cvtColor(ImagenSegmentadaColorTamanoAjustado[i], gray, COLOR_BGR2GRAY);

		// Como ya esta segmentada, cualquier píxel > 0 es parte del objeto.
		// Creamos la mascara binaria directamente de los pixeles no negros.
		threshold(gray, bw, 0, 255, THRESH_BINARY | THRESH_OTSU);

		hs = obtenerMedianaMediaHS(ImagenSegmentadaColorTamanoAjustado[i]);
		rgb = obtenerMedianaMediaRGB(ImagenSegmentadaColorTamanoAjustado[i]); // Caracteristicas de color
		hu = obtenerMomentosHu(bw); // Caracteristicas de forma (Momentos de Hu)
		props = obtenerPropiedadesImagen(bw); // Propiedades geometricas
		rugosidad = obtenerRugosidad(bw, gray);

		vectorCaracteristicas.insert(vectorCaracteristicas.end(), rgb.begin(), rgb.end());
		vectorCaracteristicas.insert(vectorCaracteristicas.end(), hs.begin(), hs.end());
		vectorCaracteristicas.insert(vectorCaracteristicas.end(), hu.begin(), hu.end());
		vectorCaracteristicas.insert(vectorCaracteristicas.end(), props.begin(), props.end());
		vectorCaracteristicas.push_back(rugosidad);
		vector_final.push_back(vectorCaracteristicas);
	}
	
	// Emision del resultado
	emit ListaCaracterisiticas(vector_final);
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
	double area = 0, perimetro = 0, circularidad = 0, relacionAspecto = 0, extension = 0, solidez = 0;
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
		// 2. Relación de Aspecto (Ancho / Alto)
		Rect rect = boundingRect(contours[idxA]);
		if (rect.height > 0) {
			relacionAspecto = (double)rect.width / rect.height;
		}
		// 3. Extensión (Área objeto / Área del rectángulo contenedor)
		double areaRect = rect.width * rect.height;
		if (areaRect > 0) {
			extension = area / areaRect;
		}
		vector<Point> hull;
		convexHull(contours[idxA], hull);
		double areaHull = contourArea(hull);
		double solidez = (areaHull > 0) ? area / areaHull : 0;

	}

	// Vector de salida
	return { area, perimetro, circularidad, relacionAspecto, extension, solidez};
}

vector<double> ExtraccionCaracteristicas::obtenerMedianaMediaHS(const Mat& ImagenSegmentadaColor) {

	// 1. Convertir la imagen de BGR a HSV
	Mat hsv;
	cvtColor(ImagenSegmentadaColor, hsv, COLOR_BGR2HSV);

	// Estructuras para H y S
	vector<double> H_f, S_f;

	// Acumuladores para la media
	double sumH = 0, sumS = 0;
	int count = 0;

	// En OpenCV: 
	// H va de 0 a 180 (para caber en un uchar)
	// S va de 0 a 255
	// V va de 0 a 255

	for (int i = 0; i < hsv.rows; i++) {
		const Vec3b* pixelHSV = hsv.ptr<Vec3b>(i);
		const Vec3b* pixelBGR = ImagenSegmentadaColor.ptr<Vec3b>(i);

		for (int j = 0; j < hsv.cols; j++) {
			// Usamos la imagen original para detectar el fondo (negro)
			if (pixelBGR[j][0] | pixelBGR[j][1] | pixelBGR[j][2]) {

				// Extraemos valores y normalizamos si lo deseas
				// H: lo dividimos por 180 para tener rango 0-1
				// S: lo dividimos por 255 para tener rango 0-1
				double h = pixelHSV[j][0] / 180.0;
				double s = pixelHSV[j][1] / 255.0;
				double v = pixelHSV[j][2] / 255.0;

				// Opcional: Filtrar pixeles muy oscuros (V < 0.1) o muy pálidos (S < 0.1)
				// para evitar ruido en el cálculo del tono (H)
				if (v > 0.1 && s > 0.05) {
					H_f.push_back(h);
					S_f.push_back(s);
					sumH += h;
					sumS += s;
					count++;
				}
			}
		}
	}

	// Lambda para calcular la mediana (mantenemos tu lógica eficiente)
	auto calcularMediana = [](vector<double>& valores) {
		if (valores.empty()) return 0.0;
		size_t mid = valores.size() / 2;
		nth_element(valores.begin(), valores.begin() + mid, valores.end());
		if (valores.size() % 2 == 0) {
			auto itMax = max_element(valores.begin(), valores.begin() + mid);
			return (*itMax + valores[mid]) / 2.0;
		}
		else {
			return valores[mid];
		}
		};

	double H_mediana = calcularMediana(H_f);
	double S_mediana = calcularMediana(S_f);

	double H_media = count ? sumH / count : 0.0;
	double S_media = count ? sumS / count : 0.0;

	// Retornamos 4 características: Mediana y Media de Hue y Saturation
	return { H_mediana, S_mediana, H_media, S_media };
}

double ExtraccionCaracteristicas::obtenerRugosidad(const Mat& BW, const Mat& gray) {
	Scalar media, desviacion;
	meanStdDev(gray, media, desviacion, BW);
	return desviacion.val[0];
}

// Esta funcion es solo para extraer las caracteristicas de todas las imagenes para el clasificador
void ExtraccionCaracteristicas::ExtraerXyGClasificacion() {
	vector<vector<double>> X;
	vector<double> G;
	QDir directory("imagenes/imagenesSegmentadas/");

	QStringList filters;
	filters << "*.jpg" << "*.png" << "*.jpeg" << "*.tif";
	QStringList files = directory.entryList(filters, QDir::Files);

	for (int i = 0; i < files.size(); ++i) {
		QString fileName = files[i];
		int numero = fileName.section('_', 0, 0).toInt();

		Mat I = imread(directory.absoluteFilePath(fileName).toStdString());
		
		vector<double> caracteristicas = ExtraerCaracteristicasImagen1(I);
	
		X.push_back(caracteristicas);
		G.push_back(numero);
	}

	// Guardar X y G en archivos CSV
	ofstream file("datos/PropsRGBHS17/X.csv");
	for (const auto& fila : X) {
		for (size_t i = 0; i < fila.size(); ++i) {
			file << fila[i] << (i == fila.size() - 1 ? "" : ",");
		}
		file << "\n";
	}
	file.close();
	ofstream fileG("datos/PropsRGBHS17/G.csv");
	for (const auto& valor : G) {
		fileG << valor << "\n"; // Un valor por línea
	}
	fileG.close();
}

vector<double> ExtraccionCaracteristicas::ExtraerCaracteristicasImagen1(const Mat& ImagenSegmentadaColorTamanoAjustado) {


	
		// Variables
		Mat gray, bw;

		// Extraccion de caracteristicas por bloques
		vector<double> vectorCaracteristicas, rgb, hu, props, hs;
		double rugosidad;
		// Convertimos la imagen segmentada a gris
		cvtColor(ImagenSegmentadaColorTamanoAjustado, gray, COLOR_BGR2GRAY);

		// Como ya esta segmentada, cualquier píxel > 0 es parte del objeto.


		// Creamos la mascara binaria directamente de los pixeles no negros.
		threshold(gray, bw, 0, 255, THRESH_BINARY | THRESH_OTSU);

		rgb = obtenerMedianaMediaRGB(ImagenSegmentadaColorTamanoAjustado); // Caracteristicas de color
		hs = obtenerMedianaMediaHS(ImagenSegmentadaColorTamanoAjustado);
		hu = obtenerMomentosHu(bw); // Caracteristicas de forma (Momentos de Hu)
		props = obtenerPropiedadesImagen(bw); // Propiedades geometricas
		rugosidad = obtenerRugosidad(bw, gray);

		vectorCaracteristicas.insert(vectorCaracteristicas.end(), rgb.begin(), rgb.end());
		vectorCaracteristicas.insert(vectorCaracteristicas.end(), hs.begin(), hs.end());
		vectorCaracteristicas.insert(vectorCaracteristicas.end(), hu.begin(), hu.end());
		vectorCaracteristicas.insert(vectorCaracteristicas.end(), props.begin(), props.end());
		vectorCaracteristicas.push_back(rugosidad);


		return vectorCaracteristicas;
}