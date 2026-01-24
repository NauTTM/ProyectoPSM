#include "ProyectoPSM.h"

ProyectoPSM::ProyectoPSM(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::ProyectoPSM)
{
    // Inicializamos la ventna principal y cargarmos interfaz grafica
    ui->setupUi(this);

    // Inicializamos los parametros del sistema y creacion de hilos
    temporizador = new QTimer(this);
    camaraThread = new QThread(this);

    // Creamos tres hilos cada uno para diferentes partes
    camara = new CVideoAcquisition();

    // mover el trabajador al hilo
    camara->moveToThread(camaraThread);

    // Arrancamos los hilos
    camaraThread->start();

    camara->SetCameraExposure(33000);
    camara->SetAutoGain();

    PrepararPestanaCapturaImagen();
    PrepararPestanaEntrenamiento();
    PrepararPestanaClasificacion();

  

    //extraccionCaracteristicas->ExtraerXyGClasificacion(a);
    //clasificadorModelo->Clasificador_RF();
   /* Mat img_ref_color, img_actual_color, img_ref_bin, img_actual_bin;
    img_actual_color = imread("imagenes/imagenesSegmentadasColor/01_000_70_003_segColor.jpg");
    
	orientacion = new Orientacion();
    img_ref_bin = imread("imagenes/imagenesSegmentadasSinResize/01_000_70_003_bw.jpg");
    img_actual_bin = imread("imagenes/imagenesSegmentadasSinResize/01_000_70_003_bw.jpg");
    orientacion->CalcularOrientacion(1, img_actual_color, img_actual_bin);*/

}

ProyectoPSM::~ProyectoPSM()
{
    camaraThread->quit();
    camaraThread->wait();
    temporizador->stop();
    threadSegmentacion->deleteLater();
	extraccionThread->deleteLater();
	clasificadorThread->deleteLater();
	orientacionThread->deleteLater();
    clasificadorModeloThread->deleteLater();
    delete ui;
}

// Funcion para el control de grabacion
void ProyectoPSM::iniciarDetenerGrabacion()
{
    // Error si no hay camara conectada
    if (!camara->CameraOK) {
        QMessageBox::warning(this, "Error", "No se pudo abrir la cámara.");
        return;
    }
    if(ui->opciones->currentWidget() == ui->Clasificacion) {
        IniciarDetenerGrabacionPestanaClasificacion();
       
    } else if(ui->opciones->currentWidget() == ui->CapturaImagen) {
        IniciarDetenerGrabacionPestanaCaptura();
	}
}

// Funcion para actualizacion de frames
void ProyectoPSM::actualizarFrame(const Mat& nuevaImagen)
{
    static int saltarFrame = 0;

    if (nuevaImagen.empty()) return;

    if (++saltarFrame % 2 != 0) return;

    if (ui->opciones->currentWidget() == ui->Clasificacion) {
        frameActual = nuevaImagen.clone();
        ui->ImagenCapturaBDD->clear();
        ActualizarFramePestanaClasificacion();
    }
    else if (ui->opciones->currentWidget() == ui->CapturaImagen) {
		frameActualCaptura = nuevaImagen.clone();
        ui->ImagenCapturaClasificacion->clear();
        ActualizarFramePestanaCaptura();
    }

}

// Funcion para captura de imagenes
void ProyectoPSM::capturarImagen()
{
    // Error si no existe imagen
    
    QString nombre;
    if (ui->opciones->currentWidget() == ui->Clasificacion) {
        if (frameActual.empty()) {
            QMessageBox::warning(this, "Captura", "No hay imagen disponible.");
            return;
        }
        nombre = generarNombreArchivoPestanaClasificacion();
        imwrite(nombre.toStdString(), frameActual);
    }
    else if (ui->opciones->currentWidget() == ui->CapturaImagen) {
        if (frameActualCaptura.empty()) {
            QMessageBox::warning(this, "Captura", "No hay imagen disponible.");
            return;
        }
        nombre = generarNombreArchivoCaptura();
        imwrite(nombre.toStdString(), frameActualCaptura);
    }

    
    QMessageBox::information(this, "Imagen guardada", nombre);
}

QImage ProyectoPSM::matToQImage(const Mat& mat)
{
    if (mat.empty()) return QImage();

    // Si la imagen es a color
    if (mat.type() == CV_8UC3) {
        static Mat rgb;
        cvtColor(mat, rgb, COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    }
    // Si es a escala de grises
    else if (mat.type() == CV_8UC1) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
    }
    else {
        return QImage();
    }

}

#pragma region Pestana Clasificacion

void ProyectoPSM::PrepararPestanaClasificacion() {

    Recording = false;
    MostrarTexto = false;

    threadSegmentacion = new QThread(this);
    extraccionThread = new QThread(this);
    clasificadorThread = new QThread(this);
    orientacionThread = new QThread(this);

    segmentacion = new Segmentacion(); // Hilo segmentacion
    extraccionCaracteristicas = new ExtraccionCaracteristicas(); // Hilo extraccion de caracteristicas
    clasificacionImagen = new ClasificacionImagen(); // Hilo clasificiacion de imagen
    orientacion = new Orientacion(); // Hilo orientacion

    segmentacion->moveToThread(threadSegmentacion);
    extraccionCaracteristicas->moveToThread(extraccionThread);
    clasificacionImagen->moveToThread(clasificadorThread);
    orientacion->moveToThread(orientacionThread);

    threadSegmentacion->start();
    extraccionThread->start();
    clasificadorThread->start();
    orientacionThread->start();

    // Iniamos el parametro para contar los frames
    ContadorFrames = 0;

    // Conectar señales y slots
    connect(ui->btnIniciarParar, SIGNAL(clicked()), this, SLOT(iniciarDetenerGrabacion()));
    connect(camara, &CVideoAcquisition::NewImageSignal, this, &ProyectoPSM::actualizarFrame);
    connect(ui->btnCapturar, SIGNAL(clicked()), this, SLOT(capturarImagen()));

    connect(this, &ProyectoPSM::enviarFrame, segmentacion, &Segmentacion::SegmentarImagen, QueuedConnection);
    connect(segmentacion, &Segmentacion::SegmentacionCompletada, this, &ProyectoPSM::MostrarImagenSegmentada);
    connect(segmentacion, &Segmentacion::SegmentacionCompletada, extraccionCaracteristicas, &ExtraccionCaracteristicas::ExtraerCaracteristicasImagen, QueuedConnection);

    connect(extraccionCaracteristicas, &ExtraccionCaracteristicas::ListaCaracterisiticas, clasificacionImagen, &ClasificacionImagen::Clasificacion, QueuedConnection);

    connect(segmentacion, &Segmentacion::SegmentacionCompletada, orientacion, &Orientacion::GuardarImagenesSegmentadas, QueuedConnection);

    connect(clasificacionImagen, &ClasificacionImagen::ResultadoClasificacion, orientacion, &Orientacion::CalcularOrientacion);

    connect(orientacion, &Orientacion::OrientacionCalculada, this, &ProyectoPSM::MostrarClase);

    // Crear carpeta dataset si no existe
    QDir dir;
    if (!dir.exists("capturas")) dir.mkdir("capturas");

}

void ProyectoPSM::IniciarDetenerGrabacionPestanaClasificacion() {
    if (RecordingCaptura) {
        QMessageBox::warning(this, "Error", "Primero detenga la grabación de captura.");
        return;
    }
    if (!Recording) {
        ui->btnIniciarParar->setStyleSheet("background-color: #E05334");
        ui->btnIniciarParar->setText("Parar");
        camara->StartStopCapture(true);
        //temporizador->start(10);
        Recording = true;
    }
    else {
        ui->btnIniciarParar->setStyleSheet("background-color: white");
        ui->btnIniciarParar->setText("Iniciar");
        camara->StartStopCapture(false);
        //temporizador->stop();
        Recording = false;
        ContadorFrames = 0;
    }
}


void ProyectoPSM::ActualizarFramePestanaClasificacion() {
    ContadorFrames++;

    Mat frameMostrar;

    //Convertimos a QImage para mostrar
    if (!BordesActuales.empty()) {
        frameMostrar = frameActual.clone();
        drawContours(frameMostrar, BordesActuales, -1, Scalar(255, 255, 0), 2);
        if (MostrarTexto && BordesActuales.size() == TiposClase.size()) {
            for (int i = 0; i < BordesActuales.size(); i++) {
                // --- 1. CÁLCULO DE POSICIONES ---
                Moments m = moments(BordesActuales[i]);
                if (m.m00 == 0) continue;
                Point centroide(m.m10 / m.m00, m.m01 / m.m00);
                Centroide = centroide;
                // --- 2. CÁLCULO DEL VECTOR DE ORIENTACIÓN ---
                double longitud = 150.0; // Largo de la flecha en píxeles
                // Convertimos el ángulo final a radianes
                // Nota: Restamos 90 si tu ángulo 0° es horizontal y quieres que apunte al frente
                double anguloDesfasado = OrientacionActual[i] + 45.0 - 180;
                double anguloRad = anguloDesfasado * CV_PI / 180.0;

                Point puntoFinal(
                    centroide.x + static_cast<int>(longitud * cos(anguloRad)),
                    centroide.y - static_cast<int>(longitud * sin(-anguloRad)) // "-" porque Y crece hacia abajo
                );
                PuntoFinal = puntoFinal;

                // --- 3. DIBUJO DE LA FLECHA ---
                arrowedLine(frameMostrar, Centroide, PuntoFinal, Scalar(255, 255, 0), 3, 8, 0, 0.3);
                QString textoClase = QString("%1").arg(TiposClase[i], 2, 10, QChar('0'));
                QString orientacionTexto = QString::number(OrientacionActual[i]);

                // 1. Definir parámetros comunes
                int fuente = FONT_HERSHEY_SIMPLEX;
                double escala = 2.0;
                int grosor = 2;
                int baseline = 0;
                Rect rect = boundingRect(BordesActuales[i]);
                // Preparar los strings
                string strClase = "Clase: " + textoClase.toStdString();
                string strOrien = "Orientacion: " + orientacionTexto.toStdString();

                // 2. Calcular tamaños de los textos
                Size tamClase = getTextSize(strClase, fuente, escala, grosor, &baseline);
                Size tamOrien = getTextSize(strOrien, fuente, escala, grosor, &baseline);

                // Usamos el ancho del más largo para las validaciones de borde derecho
                int anchoMax = max(tamClase.width, tamOrien.width);
                int altoTotal = tamClase.height + tamOrien.height + 20; // 20px de separación entre líneas

                // 3. Ajustar X (Evitar que se salga por la derecha)
                int xFinal = rect.x;
                if (xFinal + anchoMax > frameMostrar.cols) {
                    xFinal = frameMostrar.cols - anchoMax - 10; // Margen de 10px
                }
                if (xFinal < 0) xFinal = 10; // Evitar margen izquierdo negativo

                // 4. Ajustar Y (Evitar que se salga por arriba)
                // Intentamos ponerlo arriba del objeto, pero si no hay espacio (altoTotal + margen), lo ponemos debajo
                int yBase = rect.y - 10;

                if (yBase - altoTotal < 0) {
                    // No cabe arriba, lo ponemos debajo del objeto
                    yBase = rect.y + rect.height + tamClase.height + 10;
                }

                // 5. Dibujar (Orientación debajo de Clase)
                Point posClase(xFinal, yBase - tamOrien.height - 10);
                Point posOrien(xFinal, yBase);

                // Dibujar Clase
                putText(frameMostrar, strClase, posClase, fuente, escala, Scalar(255, 255, 0), grosor);

                // Dibujar Orientación
                putText(frameMostrar, strOrien, posOrien, fuente, escala, Scalar(255, 255, 0), grosor);
            }

        }
    }
    else {
        frameMostrar = frameActual;
    }
    QImage imagen = matToQImage(frameMostrar);
    frameMostrar.release();
    ui->ImagenCapturaClasificacion->setPixmap(QPixmap::fromImage(imagen));
    if (ContadorFrames % 15 == 0 || ContadorFrames == 1)
        emit enviarFrame(frameActual);

}


// Crear nombre unico segun fecha y hora
QString ProyectoPSM::generarNombreArchivoPestanaClasificacion()
{
    string extension = ".jpg";
    return QString("capturas/captura_%1%2")
        .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"))
        .arg(extension);
}

// Visualizacion de la imagen segmentada
void ProyectoPSM::MostrarImagenSegmentada(const vector<Mat>& img1, const vector<vector<Point>>& Bordes)
{
    vector<vector<Point>> bordesFiltrados;
    double areaMinima = 1300.0; 

    for (const auto& contorno : Bordes) {
        if (contourArea(contorno) > areaMinima) {
            bordesFiltrados.push_back(contorno);
        }
    }

    // Actualizar con los bordes que pasaron el filtro
    BordesActuales = bordesFiltrados;

    // Limpieza de imagenes al inicio 
    ui->ImagenSegmentada1->clear();
    ui->ImagenSegmentada2->clear();
    ui->ImagenSegmentada3->clear();

    if (img1.size() > 0) {
        QImage imagen = matToQImage(img1[0]);
        ui->ImagenSegmentada1->setPixmap(QPixmap::fromImage(imagen));
    }
    if (img1.size() > 1) {
        QImage imagen = matToQImage(img1[1]);
        ui->ImagenSegmentada2->setPixmap(QPixmap::fromImage(imagen));
    }
    if (img1.size() > 2) {
        QImage imagen = matToQImage(img1[2]);
        ui->ImagenSegmentada3->setPixmap(QPixmap::fromImage(imagen));
    }

}

// Mostrar el codigo detectadoen forma de dos digitos
void ProyectoPSM::MostrarClase(vector<int> tipoClase, vector<int> orientacion) {
    if (tipoClase.size() == 0) return;
    MostrarTexto = true;
    TiposClase = tipoClase;
    OrientacionActual = orientacion;
}

#pragma endregion

#pragma region Pestana Captura de imagen

void ProyectoPSM::PrepararPestanaCapturaImagen()
{
    QDir dir;
    if (!dir.exists("dataset")) dir.mkdir("dataset");

    RecordingCaptura = false;

    InicializarCombos();

    connect(ui->btnIniciarParar_2, SIGNAL(clicked()), this, SLOT(iniciarDetenerGrabacion()));
    connect(ui->btnCapturar_2, SIGNAL(clicked()), this, SLOT(capturarImagen()));
}

void ProyectoPSM::InicializarCombos()
{
    QStringList codigos = { "01","02","03", "04", "05", "06", "07", "08", "09", "10", "11", "12" };
    ui->comboCodigo->addItems(codigos);

    QStringList azimuth = { "000","045","090","135","180","225","270","315" };
    ui->comboAzimuth->addItems(azimuth);

    QStringList elevacion = { "000","030","060","090" };
    ui->comboElev->addItems(elevacion);

    QStringList secuencias = { "001","002","003","004","005" };
    ui->comboSeq->addItems(secuencias);
}

void ProyectoPSM::IniciarDetenerGrabacionPestanaCaptura()
{
    if (Recording) {
        QMessageBox::warning(this, "Error", "Primero detenga la grabación principal.");
        return;
    }

    if (!RecordingCaptura) {
        ui->btnIniciarParar_2->setStyleSheet("background-color: #E05334");
        ui->btnIniciarParar_2->setText("Parar");
        camara->SetCameraExposure(33000);
		camara->SetAutoGain();
        camara->StartStopCapture(true);
        //temporizador->start(0);
        RecordingCaptura = true;
    }
    else {
        ui->btnIniciarParar_2->setStyleSheet("background-color: white");
        ui->btnIniciarParar_2->setText("Iniciar");
        camara->StartStopCapture(false);
        //temporizador->stop();
        RecordingCaptura = false;
    }
}

void ProyectoPSM::ActualizarFramePestanaCaptura()
{
    QImage imagen = matToQImage(frameActualCaptura);
    ui->ImagenCapturaBDD->setPixmap(QPixmap::fromImage(imagen));
}

QString ProyectoPSM::generarNombreArchivoCaptura()
{
    QString codigo = ui->comboCodigo->currentText();
    QString az = ui->comboAzimuth->currentText();
    QString el = ui->comboElev->currentText();
    QString seq = ui->comboSeq->currentText();
    return QString("dataset/%1_%2_%3_%4.png").arg(codigo, az, el, seq);
}

#pragma endregion

#pragma region Pestana Entrenamiento

void ProyectoPSM::PrepararPestanaEntrenamiento()
{
    clasificadorModelo = new Clasificador();
    clasificadorModeloThread = new QThread(this);

    clasificadorModelo->moveToThread(clasificadorModeloThread);
    clasificadorModeloThread->start();

    ui->progressBar->setRange(0, 100); // Rango de 0% a 100%
    ui->progressBar->setValue(0);      // Inicializar en 0

    connect(ui->EntrenarClasificador, &QPushButton::clicked, clasificadorModelo, &Clasificador::Clasificador_RF);
    connect(clasificadorModelo, &Clasificador::ProgresoActualizado, ui->progressBar, &QProgressBar::setValue);
    connect(clasificadorModelo, &Clasificador::EntrenamientoFinalizado, this, &ProyectoPSM::MostrarResultadoEntrenamiento);
}


void ProyectoPSM::MostrarResultadoEntrenamiento(double precision) {
    ui->precisionLbl->setText(QString("Precision: %1%").arg(precision * 100, 0, 'f', 2));
    QMessageBox::information(this, "Entrenamiento completado", QString("Precision del clasificador: %1%").arg(precision * 100, 0, 'f', 2));

}

#pragma endregion

