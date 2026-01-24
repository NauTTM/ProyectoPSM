#include "Orientacion.h"

Orientacion::Orientacion() {
	BaseDatos = CargarReferenciasCSV("datos/Orientacion.csv");
}

void Orientacion::GuardarImagenesSegmentadas(const vector<Mat>& Imagen, const vector<vector<Point>>& Bordes, const vector<Mat> ImagenesSegmentadasBinarias, const vector<Mat> ImagenesBinariasAColor){
	ImgColor = ImagenesBinariasAColor;
	ImgBin = ImagenesSegmentadasBinarias;
}

void Orientacion::CalcularOrientacion(const vector<int> id_clase) {
    vector<int> angulo_final_vec;
    for(int i = 0; i< id_clase.size(); i++)
    {
        double angulo_base = DeterminarOrientacionTotal(id_clase[i], ImgColor[i], ImgBin[i], BaseDatos);
        int angulo_final = static_cast<int>(std::round(angulo_base / 45.0) * 45) % 360;
        angulo_final_vec.push_back(angulo_final);
    }

    emit OrientacionCalculada(id_clase,angulo_final_vec);
}

vector<Orientacion::ModeloReferencia> Orientacion::CargarReferenciasCSV(string path) {
    vector<ModeloReferencia> lista;
    ifstream file(path);
    string line;

    if (!file.is_open()) {
        cout << "Error al abrir el archivo CSV" << endl;
        return lista;
    }

    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string item;
        Orientacion::ModeloReferencia ref;

        // 1. Leer id_clase (separado por coma)
        getline(ss, item, ',');
        ref.id_clase = stoi(item);

        // 2. Leer elevacion (separado por coma)
        getline(ss, item, ',');
        ref.elevacion = stoi(item);

        // 3. Leer el resto de la línea (la signatura con espacios)
        getline(ss, item);
        stringstream ssSig(item);
        double val;
        while (ssSig >> val) {
            ref.signatura.push_back(val);
        }

        lista.push_back(ref);
    }

    file.close();
    cout << "Se cargaron " << lista.size() << " referencias correctamente." << endl;
    return lista;
}
void Orientacion::CalcularTodasSignaturas() {

    vector<ModeloReferencia> modeloRef;
    

        vector<vector<double>> X;
        vector<double> G;
        QDir directory("imagenes/imagenesSegmentadasSinResize/");

        QStringList filters;
        filters << "*.jpg" << "*.png" << "*.jpeg" << "*.tif";
        QStringList files = directory.entryList(filters, QDir::Files);
        modeloRef.resize(files.size());
        
        for (int i = 0; i < files.size(); ++i) {

            QString fileName = files[i];
            int numero = fileName.section('_', 0, 0).toInt();
			int elevacion = fileName.section('_', 2, 2).toInt();
            int orientacion = fileName.section('_', 1, 1).toInt();
            if (orientacion == 0 && elevacion != 10) {
                Mat I = imread(directory.absoluteFilePath(fileName).toStdString());

                modeloRef[i].id_clase = numero;
                modeloRef[i].elevacion = elevacion;
                modeloRef[i].signatura = ObtenerSignatura(I);
            }
        }

        ofstream fileG("datos/PropsRGBHS17/Orientacion.csv");

        if (fileG.is_open()) {
            // Opcional: Escribir cabecera del CSV
            fileG << "id_clase,elevacion,signatura\n";

            for (const auto& ref : modeloRef) {
                // 1. Guardar id y elevación
                fileG << ref.id_clase << "," << ref.elevacion << ",";

                // 2. Guardar la signatura (vector) separada por espacios o punto y coma
                // para no romper la estructura de columnas del CSV
                for (size_t j = 0; j < ref.signatura.size(); ++j) {
                    fileG << ref.signatura[j];
                    if (j < ref.signatura.size() - 1) {
                        fileG << " "; // Separador interno para el vector
                    }
                }
                fileG << "\n"; // Fin de la línea para esta referencia
            }
            fileG.close();
        }
}

double Orientacion::DeterminarOrientacionTotal(const int id_clase, const Mat& imgColor, const Mat& imgBin, const vector<ModeloReferencia>& baseDatos) {
    // ... (Pasos 1 a 3 iguales: Signatura y Match de Base de Datos)
    vector<double> sigActual = ObtenerSignatura(imgBin);
    vector<double> sigNorm = sigActual;
    NormalizarSignatura(sigNorm);
    ModeloReferencia mejorRef = EncontrarMejorClase(id_clase, sigNorm, baseDatos);
    int shift = CalcularDesplazamientoCircular(sigNorm, mejorRef.signatura);

    // 4. Preparar imagen HSV para la referencia de color
    Mat imgHSV;
    cvtColor(imgColor, imgHSV, COLOR_BGR2HSV);

    // 5. Calcular centroide
    Moments m = moments(imgBin, true);
    Point2f centroide(m.m10 / m.m00, m.m01 / m.m00);

    // 6. Verificar Referencia de Color (Punto H más alto)
    bool refArriba = ObtenerReferenciaHue(imgHSV, imgBin, centroide);

    double anguloFinal = static_cast<double>(shift);

    // 7. Lógica de corrección de 180 grados
    // Asumimos que en tus referencias el "punto de color" siempre se guardó ARRIBA
    if (!refArriba) {
        // Si el punto de color está abajo, la pieza está invertida respecto a la referencia
        anguloFinal = fmod(anguloFinal + 180.0, 360.0);
    }

    return anguloFinal;
}

vector<double> Orientacion::ObtenerSignatura(const Mat& BW) {
    vector<double> signature(360, 0.0);
    Mat s_norm;

    // 1. Asegurar formato 8-bit de un solo canal
    if (BW.channels() > 1) {
        cvtColor(BW, s_norm, COLOR_BGR2GRAY);
    }
    else {
        s_norm = BW;
    }

    // Convertir a 8 bits solo si no lo es (evita saturación si ya era 8-bit)
    if (s_norm.depth() != CV_8U) {
        s_norm.convertTo(s_norm, CV_8UC1, 255.0);
    }

    // 2. Binarización clara
    Mat imageFilled;
    // Si la imagen ya es binaria, THRESH_OTSU a veces falla si no hay varianza.
    // Es más seguro un umbral fijo si ya segmentaste antes.
    threshold(s_norm, imageFilled, 127, 255, THRESH_BINARY);

    // 3. Verificación de seguridad antes de findContours
    if (imageFilled.empty()) return signature;

    vector<vector<Point>> contours;
    // IMPORTANTE: imageFilled DEBE ser tipo 0 (CV_8UC1)
    findContours(imageFilled, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    // 2. Obtener Centroide (regionprops 'Centroid')
    Moments m = moments(imageFilled, true);
    if (m.m00 == 0) return signature; // Imagen vacía
    Point2f centro(m.m10 / m.m00, m.m01 / m.m00);

    // 3. Obtener el contorno más largo (bwboundaries 'noholes')
    if (contours.empty()) return signature;
    auto itMax = max_element(contours.begin(), contours.end(),
        [](const vector<Point>& a, const vector<Point>& b) {
            return a.size() < b.size();
        });
    vector<Point> contorno = *itMax;

    // 4. Calcular Coordenadas Polares y agrupar por ángulo
    // Usamos un vector de vectores para manejar múltiples puntos por grado
    vector<vector<double>> rhos_por_grado(360);

    for (const auto& pt : contorno) {
        // CORRECCIÓN 1: Invertir Y (MATLAB: y = -(row - centro_y)
        double dx = pt.x - centro.x;
        double dy = -(pt.y - centro.y);

        double rho = sqrt(dx * dx + dy * dy);
        double theta = atan2(dy, dx) * 180.0 / CV_PI; // Rango [-180, 180]

        // CORRECCIÓN 2: Mapear a [0, 359]
        int angleIdx = static_cast<int>(round(theta));
        if (angleIdx < 0) angleIdx += 360;
        if (angleIdx >= 360) angleIdx = 0;

        rhos_por_grado[angleIdx].push_back(rho);
    }

    // 5. Tomar el máximo para cada ángulo
    for (int i = 0; i < 360; ++i) {
        if (!rhos_por_grado[i].empty()) {
            signature[i] = *max_element(rhos_por_grado[i].begin(), rhos_por_grado[i].end());
        }
    }

    // 6. Rellenar huecos (fillmissing 'linear')
    // Interpolación simple para valores en 0
    for (int i = 0; i < 360; ++i) {
        if (signature[i] == 0) {
            int prev = (i - 1 + 360) % 360;
            int next = (i + 1) % 360;
            // Búsqueda básica de vecinos no cero
            while (signature[next] == 0 && next != i) next = (next + 1) % 360;
            if (next != i) {
                signature[i] = (signature[prev] + signature[next]) / 2.0;
            }
        }
    }

    // 7. Suavizado (smoothdata 'gaussian')
    // Usamos un filtro de caja (GaussianBlur 1D aproximado)
   vector<double> smoothed(360);
    int kernelSize = 10;
    for (int i = 0; i < 360; ++i) {
        double sum = 0;
        for (int k = -kernelSize / 2; k <= kernelSize / 2; ++k) {
            sum += signature[(static_cast<std::vector<double, allocator<double>>::size_type>(i) + k + 360) % 360];
        }
        smoothed[i] = sum / (kernelSize + 1);
    }
    signature = smoothed;

    // 8. Normalización
    double maxVal = *max_element(signature.begin(), signature.end());
    if (maxVal > 0) {
        for (double& val : signature) val /= maxVal;
    }

    return signature;
}

void Orientacion::NormalizarSignatura(vector<double>& sig) {
    if (sig.empty()) return;

    // 1. Calcular la media (mean)
    double sum = accumulate(sig.begin(), sig.end(), 0.0);
    double mean = sum / sig.size();

    // 2. Restar la media y calcular la norma L2 del vector resultante
    double norm_sq = 0.0;
    for (double& val : sig) {
        val -= mean;           // sig - mean(sig)
        norm_sq += val * val;  // Suma de cuadrados
    }

    double norm = sqrt(norm_sq);

    // 3. Dividir por la norma (Unit vector)
    // Evitar división por cero si el vector es constante
    if (norm > 1e-9) {
        for (double& val : sig) {
            val /= norm;
        }
    }
}

int Orientacion::CalcularDesplazamientoCircular(const vector<double>& sig_actual, const vector<double>& sig_ref) {
    int n = sig_actual.size(); // Normalmente 360

    // 1. Convertir vectores a cv::Mat (tipo float/double para DFT)
    Mat fa(1, n, CV_64F, (void*)sig_actual.data());
    Mat fr(1, n, CV_64F, (void*)sig_ref.data());

    // 2. Calcular DFT de ambas señales
    // OpenCV devuelve un formato complejo empaquetado (CCS) o complejo completo
    Mat FA, FR;
    dft(fa, FA, DFT_COMPLEX_OUTPUT);
    dft(fr, FR, DFT_COMPLEX_OUTPUT);

    // 3. Multiplicación en el dominio de la frecuencia: FA * conj(FR)
    // En MATLAB: fft(sig_actual) .* conj(fft(sig_ref))
    Mat correlacionF;
    mulSpectrums(FR, FA, correlacionF, 0, true); // 'true' aplica el conjugado al segundo elemento

    // 4. Transformada Inversa para volver al dominio del tiempo (IFFT)
    Mat correlacionT;
    idft(correlacionF, correlacionT, DFT_REAL_OUTPUT | DFT_SCALE);

    // 5. Encontrar el índice del valor máximo (equiv. a [~, shift] = max)
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(correlacionT, &minVal, &maxVal, &minLoc, &maxLoc);

    // El shift en MATLAB es shift-1 porque MATLAB empieza en 1.
    // En C++, maxLoc.x ya es el índice basado en 0.
    int angulo_base = maxLoc.x;

    return angulo_base;
}

bool Orientacion::CalcularAsimetria(const Mat& imgColor, const Mat& imgBin, Point2f centroide) {
    Mat imgGris;
    Mat s_norm;

    // 1. Asegurar formato 8-bit de un solo canal
    if (imgBin.channels() > 1) {
        cvtColor(imgBin, s_norm, COLOR_BGR2GRAY);
    }
    else {
        s_norm = imgBin;
    }

    // Convertir a 8 bits solo si no lo es (evita saturación si ya era 8-bit)
    if (s_norm.depth() != CV_8U) {
        s_norm.convertTo(s_norm, CV_8UC1, 255.0);
    }

    
    // 1. Convertir a gris si es necesario
    if (imgColor.channels() == 3) {
        cvtColor(imgColor, imgGris, COLOR_BGR2GRAY);
    }
    else {
        imgGris = imgColor.clone();
    }

    // 2. Aplicar máscara (double(imgGris) .* double(imgBin))
    // En C++ usamos copyTo con máscara para aislar la pieza
    Mat pieza;
    imgGris.copyTo(pieza, s_norm);
    pieza.convertTo(pieza, CV_64F); // Convertir a double para precisión en la suma

    int y_c = static_cast<int>(round(centroide.y));

    // Validar que el centroide esté dentro de los límites de la imagen
    if (y_c <= 0 || y_c >= pieza.rows) {
        return false;
    }

    // 3. Dividir la pieza en mitad superior e inferior usando Rect (ROI)
    // Rect(x, y, ancho, alto)
    Mat mitadSup = pieza(Rect(0, 0, pieza.cols, y_c));
    Mat mitadInf = pieza(Rect(0, y_c, pieza.cols, pieza.rows - y_c));

    // 4. Comparar la suma de intensidades
    // cv::sum devuelve un Scalar (4 canales), tomamos el primero [0]
    double sumaSup = sum(mitadSup)[0];
    double sumaInf = sum(mitadInf)[0];

    return (sumaSup > sumaInf);
}

Orientacion::ModeloReferencia Orientacion::EncontrarMejorClase(const int id_clase,const vector<double>& sigActual, const vector<ModeloReferencia>& baseDatos) {
    double mejorSimilitud = -1.0;
    ModeloReferencia mejorMatch;

    vector<Orientacion::ModeloReferencia> baseDatosFiltrada;

    for (const auto& ref : baseDatos) {
        if (ref.id_clase == id_clase && ref.elevacion != 10) {
            baseDatosFiltrada.push_back(ref);
        }
    }

    for (const auto& ref : baseDatosFiltrada) {
        // Calculamos la correlación máxima para esta elevación específica
        double similitud = CalcularCorrelacionMaxima(sigActual, ref.signatura);

        if (similitud > mejorSimilitud) {
            mejorSimilitud = similitud;
            mejorMatch = ref;
        }
    }
    return mejorMatch;
}
double Orientacion::CalcularCorrelacionMaxima(const vector<double>& sig1, const vector<double>& sig2) {
    int n = sig1.size();

    // 1. Preparar datos para FFT
    Mat s1(1, n, CV_64F, (void*)sig1.data());
    Mat s2(1, n, CV_64F, (void*)sig2.data());

    // 2. Transformada de Fourier
    Mat S1, S2;
    dft(s1, S1, DFT_COMPLEX_OUTPUT);
    dft(s2, S2, DFT_COMPLEX_OUTPUT);

    // 3. Multiplicación de espectros con conjugado (Correlación)
    Mat correlacionF;
    mulSpectrums(S1, S2, correlacionF, 0, true);

    // 4. Transformada Inversa para obtener la correlación en el dominio del tiempo
    Mat correlacionT;
    idft(correlacionF, correlacionT, DFT_REAL_OUTPUT | DFT_SCALE);

    // 5. Encontrar el valor máximo del pico
    double maxVal;
    minMaxLoc(correlacionT, nullptr, &maxVal, nullptr, nullptr);

    return maxVal;
}

bool Orientacion::ObtenerReferenciaHue(const Mat& imgHSV, const Mat& imgBin, Point2f centroide) {
    // 1. Separar los canales HSV
    vector<Mat> hsv_channels;
    split(imgHSV, hsv_channels);

    Mat H = hsv_channels[0]; // Canal Hue
    Mat S = hsv_channels[1]; // Canal Saturación
    Mat V = hsv_channels[2]; // Canal Brillo

    // 2. Crear una máscara de "píxeles válidos"
    // El canal H es ruidoso si la saturación o el brillo son muy bajos
    Mat mascaraValidos;
    threshold(S, mascaraValidos, 50, 255, THRESH_BINARY); // Solo píxeles con color
    bitwise_and(mascaraValidos, imgBin, mascaraValidos);  // Y que pertenezcan a la pieza

    // 3. Buscar el valor de H más alto en la zona válida
    double maxH;
    Point maxLoc;
    minMaxLoc(H, nullptr, &maxH, nullptr, &maxLoc, mascaraValidos);

    // 4. Determinar si la posición del H máximo está arriba o abajo del centroide
    // Nota: En OpenCV, Y crece hacia abajo.
    int y_centroide = static_cast<int>(round(centroide.y));

    // Si la posición Y del punto H máximo es menor que el centroide, está ARRIBA
    return (maxLoc.y < y_centroide);
}