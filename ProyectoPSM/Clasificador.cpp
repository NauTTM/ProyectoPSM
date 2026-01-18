#include "Clasificador.h"

Clasificador::Clasificador() {
}

// Clasificador del sistema
void Clasificador::Clasificador_RF() {

	// Cargar los datos para el entrenamiento
    vector<vector<double>> X = CargarObservacionesCSV("datos/PropsRGBHS17/X.csv");
    vector<double> G = CargarEtiquetasCSV("datos/PropsRGBHS17/G.csv");

    // Normalizacion de las caracteristicas (X)
	ParamsNormalizacion paramsNorm = NormalizarDatos(X);

	// Validacion cruzada, para evaluacion posterior
	KFoldPartition cv = CrearCVPartition(G.size(), 5);
    double loss = CalcularLossCV(paramsNorm.Xn, G, cv);

    // 2. CÁLCULO DE ACCURACY (Equivalente a accuracy = 1 - kfoldLoss)
    double accuracy = 1.0 - loss;
    qDebug() << "Accuracy por Validacion Cruzada:" << accuracy;

	// Entrenamiento del clasificador tipo: Random Forest
	Ptr<ml::RTrees> modeloRF = EntrenarRandomForest(paramsNorm.Xn, G);

	// Guardar el modelo entrenado
    modeloRF->save("datos/PropsRGBHS17/clasificador_RF.xml");
    FileStorage fs("datos/PropsRGBHS17/parametros_norm.xml", FileStorage::WRITE);
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
    int filas = X.size();
    int cols = X[0].size();
    Mat data(filas, cols, CV_32F);
    Mat responses(filas, 1, CV_32S);

    for (int i = 0; i < filas; ++i) {
        for (int j = 0; j < cols; ++j) {
            data.at<float>(i, j) = static_cast<float>(X[i][j]);
        }
        responses.at<int>(i, 0) = static_cast<int>(Y[i]);
    }

    auto modelo = ml::RTrees::create();

    // --- MEJORAS DE PARÁMETROS ---

    // 1. Más árboles (500) para mayor estabilidad
    modelo->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 500, 0.01));

    // 2. Mayor profundidad para que llegue a evaluar la geometría después del color
    modelo->setMaxDepth(20);

    // 3. Permitir divisiones más finas
    modelo->setMinSampleCount(2);

    // 4. Importancia de variables activa para diagnóstico
    modelo->setCalculateVarImportance(true);

    // 5. Ajuste de categorías (si tus IDs de clase no son correlativos)
    // Mat var_type(cols + 1, 1, CV_8U, ml::VAR_NUMERICAL);
    // var_type.at<uchar>(cols, 0) = ml::VAR_CATEGORICAL; 

    modelo->train(data, ml::ROW_SAMPLE, responses);

    return modelo;
}

double Clasificador::CalcularLossCV(const vector<vector<double>>& X, const vector<double>& G, const KFoldPartition& cv) {
    double totalError = 0;
    int totalMuestrasTest = 0;

    // Iteramos sobre cada Fold (K=5)
    for (int k = 0; k < cv.testIndices.size(); ++k) {
        vector<vector<double>> xTrain, xTest;
        vector<double> gTrain, gTest;

        // Construir conjuntos de entrenamiento y test para este Fold
        for (int idx : cv.trainIndices[k]) {
            xTrain.push_back(X[idx]);
            gTrain.push_back(G[idx]);
        }
        for (int idx : cv.testIndices[k]) {
            xTest.push_back(X[idx]);
            gTest.push_back(G[idx]);
        }

        // Entrenar modelo temporal con el set de entrenamiento del Fold
        Ptr<ml::RTrees> modeloTemporal = EntrenarRandomForest(xTrain, gTrain);

        // Probar el modelo con el set de test (el que se quedó fuera)
        for (size_t i = 0; i < xTest.size(); ++i) {
            Mat sample(1, xTest[i].size(), CV_32F);
            for (size_t j = 0; j < xTest[i].size(); ++j)
                sample.at<float>(0, j) = static_cast<float>(xTest[i][j]);

            float prediccion = modeloTemporal->predict(sample);
            if (std::abs(prediccion - gTest[i]) > 0.01) {
                totalError++;
            }
            totalMuestrasTest++;
        }
    }

    return (totalMuestrasTest > 0) ? (totalError / totalMuestrasTest) : 1.0;
}