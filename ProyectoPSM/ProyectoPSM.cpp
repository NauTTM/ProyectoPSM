#include "ProyectoPSM.h"
#include "ui_ProyectoPSM.h"


ProyectoPSM::ProyectoPSM(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::ProyectoPSM)
{
    ui->setupUi(this);
    //Mat img = imread("dataset/01_000_10_001.jpg");

    temporizador = new QTimer(this);
    Recording = false;
    threadSegmentacion = new QThread(this);
    extraccionThread = new QThread(this);
	segmentacion = new Segmentacion();
	extraccionCaracteristicas = new ExtraccionCaracteristicas();
    // mover el trabajador al hilo
    segmentacion->moveToThread(threadSegmentacion);
	extraccionCaracteristicas->moveToThread(extraccionThread);
    threadSegmentacion->start();
	extraccionThread->start();
	ContadorFrames = 0;

    // Conectar señales y slots
 //   connect(ui->btnStart, SIGNAL(clicked()), this, SLOT(iniciarDetenerGrabacion()));
 //   connect(&camara, SIGNAL(NewImageSignal()), this, SLOT(GetImage()));
 //   connect(temporizador, SIGNAL(timeout()), this, SLOT(actualizarFrame()));
 //   connect(ui->btnCapture, SIGNAL(clicked()), this, SLOT(capturarImagen()));
 //   connect(this, &ProyectoPSM::enviarFrame, segmentacion, &Segmentacion::SegmentarImagen,QueuedConnection);
	//connect(segmentacion, &Segmentacion::SegmentacionCompletada, this, &ProyectoPSM::MostrarImagenSegmentada); // eliminar linea cuando proceda
 //   connect(segmentacion, &Segmentacion::SegmentacionCompletada, extraccionCaracteristicas, &ExtraccionCaracteristicas::ExtraerCaracteristicasImagen, QueuedConnection);
    
	/*clasificador = new Clasificador();
    clasificador->Clasificador_RF();*/

    Mat img = imread("results/10_270_40_003_norm_01.png");
    ExtraccionCaracteristicas::VectorCaracteristicas caracteristicas = extraccionCaracteristicas->ExtraerCaracteristicasImagen(img);
    vector<double> caracteristicasVector =
    {
             caracteristicas.rgb.R_mediana, caracteristicas.rgb.G_mediana, caracteristicas.rgb.B_mediana,
             caracteristicas.rgb.R_media,   caracteristicas.rgb.G_media,   caracteristicas.rgb.B_media,
             caracteristicas.hu.hu1,        caracteristicas.hu.hu2,        caracteristicas.hu.hu3,
             caracteristicas.props.area,    caracteristicas.props.circularidad, caracteristicas.props.perimetro

    };

	clasificacionImagen = new ClasificacionImagen();
	clasificacionImagen->Clasificacion(caracteristicasVector);
    // Crear carpeta dataset si no existe
    QDir dir;
    if (!dir.exists("dataset")) dir.mkdir("dataset");

    
}

ProyectoPSM::~ProyectoPSM()
{
    camara.~CVideoAcquisition();
    temporizador->stop();
    threadSegmentacion->deleteLater();
	extraccionThread->deleteLater();
    delete ui;
}


void ProyectoPSM::iniciarDetenerGrabacion()
{
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

void ProyectoPSM::actualizarFrame()
{
	frameActual = camara.GetImage();

    if (frameActual.empty()) return;
	ContadorFrames++;

	Mat frameMostrar = frameActual.clone();
     //Convertimos a QImage para mostrar
	if (BordesActuales.size() > 0)
        drawContours(frameMostrar, BordesActuales, -1, Scalar(0, 255, 0), 2);
    QImage imagen = matToQImage(frameMostrar);
    ui->capturarImagen->setPixmap(QPixmap::fromImage(imagen).scaled(
        ui->capturarImagen->size(), KeepAspectRatio, SmoothTransformation));
	if (ContadorFrames % 30 == 0 || ContadorFrames == 1)
	    emit enviarFrame(frameActual);
}


void ProyectoPSM::capturarImagen()
{
    if (frameActual.empty()) {
        QMessageBox::warning(this, "Captura", "No hay imagen disponible.");
        return;
    }

    QString nombre = generarNombreArchivo();
    imwrite(nombre.toStdString(), frameActual);
    QMessageBox::information(this, "Imagen guardada", nombre);
}

QString ProyectoPSM::generarNombreArchivo()
{
    return QString("dataset/1.jpg");
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

void ProyectoPSM::MostrarImagenSegmentada(const Mat& img1, const vector<vector<Point>>& Bordes)
{
    if(Bordes.size() > 0) BordesActuales = Bordes;
    QImage imagen = matToQImage(img1);
    ui->ImagenSegmentada->setPixmap(QPixmap::fromImage(imagen).scaled(
        ui->ImagenSegmentada->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}
