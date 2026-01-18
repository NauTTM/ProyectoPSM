#include "ProyectoPSM.h"
#include "ui_ProyectoPSM.h"


ProyectoPSM::ProyectoPSM(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::ProyectoPSM)
{
    // Inicializamos la ventna principal y cargarmos interfaz grafica
    ui->setupUi(this);

    // Inicializamos los parametros del sistema y creacion de hilos
    temporizador = new QTimer(this);
    Recording = false;
    MostrarTexto = false;
    threadSegmentacion = new QThread(this);
    extraccionThread = new QThread(this);
    clasificadorThread = new QThread(this);
	orientacionThread = new QThread(this);

    // Creamos tres hilos cada uno para diferentes partes
	segmentacion = new Segmentacion(); // Hilo segmentacion
	extraccionCaracteristicas = new ExtraccionCaracteristicas(); // Hilo extraccion de caracteristicas
    clasificacionImagen = new ClasificacionImagen(); // Hilo clasificiacion de imagen
	orientacion = new Orientacion(); // Hilo orientacion
    // mover el trabajador al hilo
    segmentacion->moveToThread(threadSegmentacion);
	extraccionCaracteristicas->moveToThread(extraccionThread);
    clasificacionImagen->moveToThread(clasificadorThread);
	orientacion->moveToThread(orientacionThread);
    // Arrancamos los hilos
    threadSegmentacion->start();
	extraccionThread->start();
	clasificadorThread->start();
	orientacionThread->start();
    // Iniamos el parametro para contar los frames
	ContadorFrames = 0;
    
    // Conectar señales y slots
    connect(ui->btnStart, SIGNAL(clicked()), this, SLOT(iniciarDetenerGrabacion())); 
    connect(&camara, SIGNAL(NewImageSignal()), this, SLOT(GetImage()));
    connect(temporizador, SIGNAL(timeout()), this, SLOT(actualizarFrame()));
    connect(ui->btnCapture, SIGNAL(clicked()), this, SLOT(capturarImagen()));
    connect(this, &ProyectoPSM::enviarFrame, segmentacion, &Segmentacion::SegmentarImagen,QueuedConnection);
	connect(segmentacion, &Segmentacion::SegmentacionCompletada, this, &ProyectoPSM::MostrarImagenSegmentada); // eliminar linea cuando proceda
    connect(segmentacion, &Segmentacion::SegmentacionCompletada, extraccionCaracteristicas, &ExtraccionCaracteristicas::ExtraerCaracteristicasImagen, QueuedConnection);
	connect(extraccionCaracteristicas, &ExtraccionCaracteristicas::ListaCaracterisiticas, clasificacionImagen, &ClasificacionImagen::Clasificacion, QueuedConnection);
    

     connect(segmentacion, &Segmentacion::SegmentacionCompletada, orientacion, &Orientacion::GuardarImagenesSegmentadas, QueuedConnection);
     connect(clasificacionImagen, &ClasificacionImagen::ResultadoClasificacion, orientacion, &Orientacion::CalcularOrientacion);
     connect(orientacion, &Orientacion::OrientacionCalculada, this, &ProyectoPSM::MostrarClase);
    //extraccionCaracteristicas->ExtraerXyGClasificacion(a);
    //clasificadorModelo->Clasificador_RF();
   /* Mat img_ref_color, img_actual_color, img_ref_bin, img_actual_bin;
    img_actual_color = imread("imagenes/imagenesSegmentadasColor/01_000_70_003_segColor.jpg");
    
	orientacion = new Orientacion();
    img_ref_bin = imread("imagenes/imagenesSegmentadasSinResize/01_000_70_003_bw.jpg");
    img_actual_bin = imread("imagenes/imagenesSegmentadasSinResize/01_000_70_003_bw.jpg");
    orientacion->CalcularOrientacion(1, img_actual_color, img_actual_bin);*/


    // Crear carpeta dataset si no existe
    QDir dir;
    if (!dir.exists("capturas")) dir.mkdir("capturas");

    
}

ProyectoPSM::~ProyectoPSM()
{
    camara.~CVideoAcquisition();
    temporizador->stop();
    threadSegmentacion->deleteLater();
	extraccionThread->deleteLater();
	clasificadorThread->deleteLater();
	orientacionThread->deleteLater();
    delete ui;
}

// Funcion para el control de grabacion
void ProyectoPSM::iniciarDetenerGrabacion()
{
    // Error si no hay camara conectada
    if (!camara.CameraOK) {
        QMessageBox::warning(this, "Error", "No se pudo abrir la cámara.");
        return;
    }

    if (!Recording) {
        ui->btnStart->setStyleSheet("background-color: #E05334");
        ui->btnStart->setText("Parar");
        camara.SetCameraAutoExposure();
		camara.StartStopCapture(true);
        temporizador->start(30);            // 30 ms ~ 33 fps
		Recording = true;
    }
    else {
        ui->btnStart->setStyleSheet("background-color: white");
        ui->btnStart->setText("Iniciar");
        camara.StartStopCapture(false);
        temporizador->stop();
		Recording = false;
		ContadorFrames = 0;
    }
}

// Funcion para actualizacion de frames
void ProyectoPSM::actualizarFrame()
{
	frameActual = camara.GetImage();

    if (frameActual.empty()) return;
	ContadorFrames++;

	Mat frameMostrar = frameActual.clone();

     //Convertimos a QImage para mostrar
    if (BordesActuales.size() > 0) {
        drawContours(frameMostrar, BordesActuales, -1, Scalar(0, 255, 0), 2);
        if (MostrarTexto && BordesActuales.size() == TiposClase.size()) {
            for (int i = 0; i < BordesActuales.size(); i++) {
                Rect rect = boundingRect(BordesActuales[i]);
                Point posicionTexto(rect.x, rect.y - 50); // 10px arriba del objeto
				Point posicionOrientacion(rect.x, rect.y - 5); // 50px arriba del objeto
                QString textoClase = QString("%1").arg(TiposClase[i], 2, 10, QChar('0'));
                QString orientacionTexto = QString::number(OrientacionActual[i]);
                putText(frameMostrar,
                    "Clase: "+textoClase.toStdString(), // Convertimos QString a std::string
                    posicionTexto,
                    FONT_HERSHEY_SIMPLEX,
                    2,              // Tamaño
                    Scalar(0, 255, 0), // Rojo para que resalte
                    2);

                putText(frameMostrar,
                    "Orientacion: "+orientacionTexto.toStdString(), // Convertimos QString a std::string
                    posicionOrientacion,
                    FONT_HERSHEY_SIMPLEX,
                    2,              // Tamaño
                    Scalar(0, 255, 0), // Rojo para que resalte
                    2);
            }

        }
    }
    QImage imagen = matToQImage(frameMostrar);
    ui->capturarImagen->setPixmap(QPixmap::fromImage(imagen).scaled(
        ui->capturarImagen->size(), KeepAspectRatio, SmoothTransformation));
	if (ContadorFrames % 60 == 0 || ContadorFrames == 1)
	    emit enviarFrame(frameActual);
    frameMostrar.release();
}

// Funcion para captura de imagenes
void ProyectoPSM::capturarImagen()
{
    // Error si no existe imagen
    if (frameActual.empty()) {
        QMessageBox::warning(this, "Captura", "No hay imagen disponible.");
        return;
    }

    QString nombre = generarNombreArchivo();
    imwrite(nombre.toStdString(), frameActual);
    QMessageBox::information(this, "Imagen guardada", nombre);
}

// Crear nombre unico segun fecha y hora
QString ProyectoPSM::generarNombreArchivo()
{
	string extension = ".jpg";
    return QString("capturas/captura_%1%2")
        .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"))
        .arg(extension);
}

QImage ProyectoPSM::matToQImage(const Mat& mat)
{
    // Si la imagen es a color
    if (mat.type() == CV_8UC3) {
        Mat rgb;
        cvtColor(mat, rgb, COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
    }
    // Si es a escala de grises
    else if (mat.type() == CV_8UC1) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8).copy();
    }
    else {
        return QImage();
    }

}

// Visualizacion de la imagen segmentada
void ProyectoPSM::MostrarImagenSegmentada(const vector<Mat>& img1, const vector<vector<Point>>& Bordes)
{    
    if(Bordes.size() > 0) 
        BordesActuales = Bordes;

        QImage imagen = matToQImage(img1[0]);
        ui->ImagenSegmentada->setPixmap(QPixmap::fromImage(imagen).scaled(
            ui->ImagenSegmentada->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        if (img1.size() > 1) {
            QImage imagen = matToQImage(img1[1]);
            ui->ImagenSegmentada_2->setPixmap(QPixmap::fromImage(imagen).scaled(
                ui->ImagenSegmentada_2->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
        if (img1.size() > 2) {
            QImage imagen = matToQImage(img1[2]);
            ui->ImagenSegmentada_3->setPixmap(QPixmap::fromImage(imagen).scaled(
                ui->ImagenSegmentada_3->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }

}

// Mostrar el codigo detectadoen forma de dos digitos
void ProyectoPSM::MostrarClase(vector<int> tipoClase, vector<int> orientacion) {
    if (tipoClase.size() == 0) return;
    MostrarTexto = true;
    TiposClase = tipoClase;
    OrientacionActual = orientacion;
}