#include "Clasificador.h"

Clasificador::Clasificador() {
}

// Clasificador del sistema
void Clasificador::Clasificador_RF() {

	// Cargar los datos para el entrenamiento
    vector<vector<double>> X = CargarObservacionesCSV("datos/X.csv");
    vector<double> G = CargarEtiquetasCSV("datos/G.csv");

    // Normalizacion de las caracteristicas (X)
	ParamsNormalizacion paramsNorm = NormalizarDatos(X);

	// Validacion cruzada, para evaluacion posterior
	KFoldPartition cv = CrearCVPartition(G.size(), 5);

	// Entrenamiento del clasificador tipo: Random Forest
	Ptr<ml::RTrees> modeloRF = EntrenarRandomForest(paramsNorm.Xn, G);

	// Guardar el modelo entrenado
    modeloRF->save("datos/clasificador_RF.xml");
    FileStorage fs("datos/parametros_norm.xml", FileStorage::WRITE);
    fs << "mu" << Mat(paramsNorm.mu).t();       // Guardamos mu como fila
    fs << "sigma" << Mat(paramsNorm.sigma).t(); // Guardamos sigma como fila
    fs.release();
}

// Funcion cargar observaciones
vector<vector<double>> Clasificador::CargarObservacionesCSV(QString rutaArchivo) {
    // Inicializacion de la estructura de datos
	vector<vector<double>> X;
    QFile archivo(rutaArchivo);

    // Apertura del fichero en modo texto
    archivo.open(QIODevice::ReadOnly | QIODevice::Text);
    // Lectura
    QTextStream in(&archivo);
    while (!in.atEnd()) {
        QString linea = in.readLine();
        if (linea.isEmpty()) continue; //Filtrado de lineas vacias

        // Separacion de los valores
        QStringList valores = linea.split(',');
        vector<double> fila;

        // Conversion a valores numericos
        for (const QString& val : valores) {
            fila.push_back(val.toDouble());
        }
        // Construccion matriz X
        X.push_back(fila);
    }

    // Cierre del archivo y retorno
    archivo.close();
    return X;
}

// Funcion cargar etiquetas
vector<double> Clasificador::CargarEtiquetasCSV(QString nombreArchivo) {
    // Inicializacion de estructuras
    vector<double> G;
    QFile archivo(nombreArchivo);

    // Apertura del fichero
    archivo.open(QIODevice::ReadOnly | QIODevice::Text);
    // Lectura
    QTextStream in(&archivo);
    while (!in.atEnd()) {
        QString linea = in.readLine().trimmed(); // Limpieza de la linea
        // Filtrado de lineas vacias
        if (!linea.isEmpty()) {
            // Conversion y almacenamiento
            G.push_back(linea.toDouble());
        }
    }

    // Cierre y retorno
    archivo.close();
    return G;
}

// Funcion normalizar datos
Clasificador::ParamsNormalizacion Clasificador::NormalizarDatos(const vector<vector<double>>& X) {
    // Dimensiones del problema
    size_t numMuestras = X.size();
    size_t numFeatures = X[0].size();

    // Inicializacion de la estructura de salida
    ParamsNormalizacion res;
    res.mu.assign(numFeatures, 0.0);
    res.sigma.assign(numFeatures, 0.0);
    res.Xn = X; // Copiamos la estructura original

    // Calcular la Media (mu) por cada columna (caracteristica)
    for (size_t j = 0; j < numFeatures; ++j) {
        double suma = 0.0;
        for (size_t i = 0; i < numMuestras; ++i) {
            suma += X[i][j];
        }
        res.mu[j] = suma / numMuestras;
    }

    // Calcular la Desviacion Estandar (sigma) por cada columna (caracteristica)
    for (size_t j = 0; j < numFeatures; ++j) {
        double sumaVarianza = 0.0;
        for (size_t i = 0; i < numMuestras; ++i) {
            double diff = X[i][j] - res.mu[j];
            sumaVarianza += diff * diff;
        }
        res.sigma[j] = std::sqrt(sumaVarianza / (numMuestras - 1));

        // Evitar division por cero si la columna es constante
        if (res.sigma[j] == 0) res.sigma[j] = 1.0;
    }

    // Aplicar Normalizacio: Xn = (X - mu) ./ sigma
    for (size_t i = 0; i < numMuestras; ++i) {
        for (size_t j = 0; j < numFeatures; ++j) {
            res.Xn[i][j] = (X[i][j] - res.mu[j]) / res.sigma[j];
        }
    }

    // Retorno
    return res;
}

// Funcion crear particion
Clasificador::KFoldPartition Clasificador::CrearCVPartition(int numMuestras, int K) {
    KFoldPartition cv;

    // Crear vector de indices [0, 1, 2, ..., N-1]
    vector<int> indices(numMuestras);
    iota(indices.begin(), indices.end(), 0);

    // Barajar los indices aleatoriamente (equivalente al comportamiento de cvpartition)
    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    // Calculo del tamano de cada fold (N/K)
    int tamFold = numMuestras / K;

    // Bucle principal sobre los K folds
    for (int k = 0; k < K; ++k) {
        vector<int> testIdx;
        vector<int> trainIdx;

        // Determinar rango del fold actual
        int inicio = k * tamFold;
        int fin = (k == K - 1) ? numMuestras : (k + 1) * tamFold;

        // Separar indices en Test y Train
        for (int i = 0; i < numMuestras; ++i) {
            if (i >= inicio && i < fin) {
                testIdx.push_back(indices[i]);
            }
            else {
                trainIdx.push_back(indices[i]);
            }
        }

        // Almacenamiento de las particiones
        cv.testIndices.push_back(testIdx);
        cv.trainIndices.push_back(trainIdx);
    }

    // Retorno
    return cv;
}

// Funcion entrenar el clasificador Random Forest
Ptr<ml::RTrees> Clasificador::EntrenarRandomForest(const vector<vector<double>>& X, const vector<double>& Y) {

    // Convertir datos de std::vector a cv::Mat (Requerido por OpenCV)
    int filas = X.size();
    int cols = X[0].size();
    Mat data(filas, cols, CV_32F);
    Mat responses(filas, 1, CV_32S); // Etiquetas como enteros

    // Copia de los datos elementos a elementos
    for (int i = 0; i < filas; ++i) {
        for (int j = 0; j < cols; ++j) {
            data.at<float>(i, j) = static_cast<float>(X[i][j]);
        }
        responses.at<int>(i, 0) = static_cast<int>(Y[i]);
    }

    // Configurar el modelo Random Forest
    auto modelo = ml::RTrees::create();

    // Configuracion de los parametros del modelo
    modelo->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 0.1)); // Max de iteraciones 100, umbral de convergencia 0.1

    // 'Learners', 'Tree' (Configuración de los arboles individuales)
    modelo->setMaxDepth(10);           // Profundidad maxima
    modelo->setMinSampleCount(2);      // Muestras minimas para dividir
    modelo->setRegressionAccuracy(0);
    modelo->setCalculateVarImportance(true);

    // Entrenar
    modelo->train(data, ml::ROW_SAMPLE, responses);

    // Retorno
    return modelo;
}