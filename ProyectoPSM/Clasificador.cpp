#include "Clasificador.h"

Clasificador::Clasificador() {
}

void Clasificador::Clasificador_RF() {

	// 1. Cargar los datos
    vector<vector<double>> X = CargarObservacionesCSV("datos/X.csv");
    vector<double> G = CargarEtiquetasCSV("datos/G.csv");

    // 2. Normalizacion de X
	ParamsNormalizacion paramsNorm = NormalizarDatos(X);

	// 3. Validacion cruzada, para evaluacion posterior
	KFoldPartition cv = CrearCVPartition(G.size(), 5);

	//4. Entrenamiento del Random Forest
	Ptr<ml::RTrees> modeloRF = EntrenarRandomForest(paramsNorm.Xn, G);

    // 5. Evaluacion en MATLAB
	// 6. Guardar el modelo entrenado
    modeloRF->save("datos/clasificador_RF.xml");
    FileStorage fs("datos/parametros_norm.xml", FileStorage::WRITE);
    fs << "mu" << Mat(paramsNorm.mu).t();       // Guardamos mu como fila
    fs << "sigma" << Mat(paramsNorm.sigma).t(); // Guardamos sigma como fila
    fs.release();
}

vector<vector<double>> Clasificador::CargarObservacionesCSV(QString rutaArchivo) {
	vector<vector<double>> X;
    QFile archivo(rutaArchivo);

    archivo.open(QIODevice::ReadOnly | QIODevice::Text);
    QTextStream in(&archivo);
    while (!in.atEnd()) {
        QString linea = in.readLine();
        if (linea.isEmpty()) continue;

        QStringList valores = linea.split(',');
        vector<double> fila;

        for (const QString& val : valores) {
            fila.push_back(val.toDouble());
        }
        X.push_back(fila);
    }

    archivo.close();
    return X;
}

vector<double> Clasificador::CargarEtiquetasCSV(QString nombreArchivo) {
    vector<double> G;
    QFile archivo(nombreArchivo);

    archivo.open(QIODevice::ReadOnly | QIODevice::Text);
    QTextStream in(&archivo);
    while (!in.atEnd()) {
        QString linea = in.readLine().trimmed();
        if (!linea.isEmpty()) {
            G.push_back(linea.toDouble());
        }
    }

    archivo.close();
    return G;
}

Clasificador::ParamsNormalizacion Clasificador::NormalizarDatos(const vector<vector<double>>& X) {
    size_t numMuestras = X.size();
    size_t numFeatures = X[0].size();

    ParamsNormalizacion res;
    res.mu.assign(numFeatures, 0.0);
    res.sigma.assign(numFeatures, 0.0);
    res.Xn = X; // Copiamos la estructura original

    // 1. Calcular la Media (mu) por cada columna
    for (size_t j = 0; j < numFeatures; ++j) {
        double suma = 0.0;
        for (size_t i = 0; i < numMuestras; ++i) {
            suma += X[i][j];
        }
        res.mu[j] = suma / numMuestras;
    }

    // 2. Calcular la Desviación Estándar (sigma) por cada columna
    for (size_t j = 0; j < numFeatures; ++j) {
        double sumaVarianza = 0.0;
        for (size_t i = 0; i < numMuestras; ++i) {
            double diff = X[i][j] - res.mu[j];
            sumaVarianza += diff * diff;
        }
        // std(X, [], 1) en MATLAB usa N-1 por defecto (estimador insesgado)
        res.sigma[j] = std::sqrt(sumaVarianza / (numMuestras - 1));

        // Evitar división por cero si la columna es constante
        if (res.sigma[j] == 0) res.sigma[j] = 1.0;
    }

    // 3. Aplicar Normalización: Xn = (X - mu) ./ sigma
    for (size_t i = 0; i < numMuestras; ++i) {
        for (size_t j = 0; j < numFeatures; ++j) {
            res.Xn[i][j] = (X[i][j] - res.mu[j]) / res.sigma[j];
        }
    }

    return res;
}

Clasificador::KFoldPartition Clasificador::CrearCVPartition(int numMuestras, int K) {
    KFoldPartition cv;

    // 1. Crear vector de índices [0, 1, 2, ..., N-1]
    vector<int> indices(numMuestras);
    iota(indices.begin(), indices.end(), 0);

    // 2. Barajar los índices aleatoriamente (equivalente al comportamiento de cvpartition)
    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    int tamFold = numMuestras / K;

    for (int k = 0; k < K; ++k) {
        vector<int> testIdx;
        vector<int> trainIdx;

        // Determinar rango del fold actual
        int inicio = k * tamFold;
        int fin = (k == K - 1) ? numMuestras : (k + 1) * tamFold;

        // Separar índices en Test y Train
        for (int i = 0; i < numMuestras; ++i) {
            if (i >= inicio && i < fin) {
                testIdx.push_back(indices[i]);
            }
            else {
                trainIdx.push_back(indices[i]);
            }
        }

        cv.testIndices.push_back(testIdx);
        cv.trainIndices.push_back(trainIdx);
    }

    return cv;
}

Ptr<ml::RTrees> Clasificador::EntrenarRandomForest(const vector<vector<double>>& X, const vector<double>& Y) {

    // 1. Convertir datos de std::vector a cv::Mat (Requerido por OpenCV)
    int filas = X.size();
    int cols = X[0].size();
    Mat data(filas, cols, CV_32F);
    Mat responses(filas, 1, CV_32S); // Etiquetas como enteros

    for (int i = 0; i < filas; ++i) {
        for (int j = 0; j < cols; ++j) {
            data.at<float>(i, j) = static_cast<float>(X[i][j]);
        }
        responses.at<int>(i, 0) = static_cast<int>(Y[i]);
    }

    // 2. Configurar el modelo (Equivalente a tus parámetros de MATLAB)
    auto modelo = ml::RTrees::create();

    // 'NumLearningCycles', 100
    modelo->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 0.1));

    // 'Learners', 'Tree' (Configuración de los árboles individuales)
    modelo->setMaxDepth(10);           // Profundidad máxima
    modelo->setMinSampleCount(2);      // Muestras mínimas para dividir
    modelo->setRegressionAccuracy(0);
    modelo->setCalculateVarImportance(true);

    // 3. Entrenar
    modelo->train(data, ml::ROW_SAMPLE, responses);

    return modelo;
}